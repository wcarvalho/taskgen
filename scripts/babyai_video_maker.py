import json
import numpy as np
import os.path
from functools import partial
import torch

# ======================================================
# RLPY
# ======================================================
from rlpyt.samplers.async_.cpu_sampler import AsyncCpuSampler
from rlpyt.samplers.parallel.cpu.sampler import CpuSampler
from rlpyt.samplers.serial.sampler import SerialSampler


from rlpyt.samplers.parallel.cpu.collectors import CpuEvalCollector
from rlpyt.samplers.serial.collectors import SerialEvalCollector


from rlpyt.runners.async_rl import AsyncRlEval
from rlpyt.runners.minibatch_rl import MinibatchRlEval
from rlpyt.utils.launching.affinity import make_affinity
from rlpyt.utils.collections import AttrDict
from rlpyt.utils.buffer import numpify_buffer, buffer_func
# ======================================================
# Our
# ======================================================
# from sfgen.babyai.agent_configs import configs
from sfgen.babyai.env import BabyAIEnv
from sfgen.tools.video_maker import image_initialization, update_image, VideoMaker
from experiments.babyai_exp import load_env_setting, load_algo_agent, load_instr_preprocessor

def load_filename(path, itr=None):
    if not os.path.exists(path):
        raise RuntimeError(f"Doesn't exist: {path}")
    if os.path.isdir(path):
        if itr:
            params = os.path.join(path, f'itr_{itr}.pkl')
        else:
            params = os.path.join(path, 'params.pkl')
        settings = os.path.join(path, 'params.json')
        if os.path.exists(params) and os.path.exists(settings):
            print("="*25)
            print(f"Found params in path {params}")
            print("="*25)
            return params, settings
        else:
            raise RuntimeError(f"Counldn't find 'params.pkl' in path: {path}")
    elif os.path.isfile(path):
        raise RuntimeError(f"Please provide directory only")
    else:
        raise RuntimeError(f"Neither dir nor file: {path}")

def mission2text(mission, inverse_vocab):
    indices = [int(i.item()) for i in mission if i > 0]
    words = [inverse_vocab[i] for i in indices]
    return " ".join(words)


def make_video(trajectory, config, video_path, inverse_vocab, xlim=3, ylim=3, title_size=12, fps=1):
    obs = trajectory['env_data'].observation
    reward = trajectory['env_data'].reward


    task = mission2text(obs.mission[0,0], inverse_vocab)

    # -----------------------
    # make task fit by using multiple lines
    # -----------------------
    words = task.split(" ")
    lines = [" ".join(words[i:i+5]) for i in range(0, len(words), 5)]
    task = "\n".join(lines)

    length = obs.mission.shape[0]


    boxes = {
        'image': {'x':[0,xlim], 'y':[0,ylim]},
    }
    initialization_fns={
        'image': partial(image_initialization, 
            title=f"{task}\nlength={length}\nreward={reward.sum()}",
            title_size=title_size),
    }
    update_fns = {
        'image': update_image
    }

    video_maker = VideoMaker(
        xlim=xlim, ylim=ylim, boxes=boxes,
        fps=fps,
        settings=config,
        update_fns=update_fns,
        initialization_fns=initialization_fns,
        verbosity=1,
    )

    video_maker.make(
        env_data=trajectory['env_data'],
        agent_data=trajectory['agent_data'],
        length=length,
        video_path=video_path,
    )

def main(path,
    asynch=False,
    n_parallel=1,
    num_success=10,
    num_failure=10,
    batch_T=1000,
    trajectories=100,
    verbosity=0,
    xlim=3,
    title_size=12,
    fps=1,
    rootdir=".",
    **kwargs):

    # ======================================================
    # load settings
    # ======================================================
    pkl_file, settings_file = load_filename(path)
    with open(settings_file, 'r') as f:
        config = json.load(f)

    config['level']['rootdir'] = rootdir
    config, instr_preprocessor, task2idx, horizon = load_env_setting(config, rootdir=rootdir)

    if not torch.cuda.is_available():
        ckpt = torch.load(pkl_file, map_location=torch.device('cpu'))
    else:
        ckpt = torch.load(pkl_file)

    if config['settings']['algorithm'] in ['r2d1']:
        config['agent']['eps_init'] = config['agent']['eps_eval']
        config['agent']['eps_final'] = config['agent']['eps_eval']
        config['agent']['eps_itr_min'] = 0
        config['agent']['eps_itr_max'] = 1




    algo, agent = load_algo_agent(config,
        agent_kwargs=dict(
            initial_model_state_dict=ckpt['agent_state_dict'],
            )
        )


    # ======================================================
    # load sampler + eval class
    # ======================================================
    if asynch:
        sampler_class = AsyncCpuSampler
        runner_class = AsyncRlEval
        raise RuntimeError("not implemented")
    else:
        runner_class = MinibatchRlEval

        if n_parallel > 1:
            sampler_class = CpuSampler
            # collecter_class = CpuEvalCollector
            # raise RuntimeError("not implemented")
        else:
            sampler_class = SerialSampler
            # collecter_class = SerialEvalCollector


    # ======================================================
    # load affinity
    # ======================================================
    # affinity = make_affinity(
    #     run_slot=0,
    #     n_cpu_core=n_parallel,  # Use 16 cores across all experiments.
    #     n_gpu=0,  # Use 8 gpus across all experiments.
    #     sample_gpu_per_run=0,
    #     async_sample=False, # whether async sample/optim. irrelevant here.
    # )
    affinity=AttrDict(
            workers_cpus=list(range(n_parallel)),
            optimizer=[])

    # ======================================================
    # start up
    # ======================================================
    sampler = sampler_class(
        EnvCls=BabyAIEnv,
        env_kwargs=config['env'],
        eval_env_kwargs=config['eval_env'],
        batch_T=batch_T,
        batch_B=n_parallel,
    )

    runner = runner_class(
        algo=algo,
        agent=agent,
        sampler=sampler,
        affinity=affinity,
        **config["runner"],
    )
    runner.startup()


    # ======================================================
    # collect success + failure
    # ======================================================
    num_collected = 0
    successes = []
    failures = []
    agent.eval_mode(0)

    while (len(successes) < num_success or len(failures) < num_failure) and num_collected < trajectories:
        num_collected += 1
        # collect samples
        if verbosity:
            print("="*50)
            print("Collecting new batch")
            print("="*50)

        # hack
        runner.sampler.agent_inputs, runner.sampler.traj_infos = runner.sampler.collector.start_envs()
        samples, info = runner.sampler.obtain_samples(0)
        samples = numpify_buffer(samples)
        samples = buffer_func(samples, partial(np.array, copy=True))
        # look at starts & endings
        endings = samples.env.done[:, 0].nonzero()[0]
        starts = np.array([0] + [i.item() for i in endings[:-1]+1])

        for s, e in zip(starts, endings):
            info = dict(
                # reward=reward,
                env_data=samples.env[s:e+1],
                agent_data=samples.agent[s:e+1],
                indices=(s,e), # was useful for debugging
                )
            reward = samples.env.reward[s:e+1]
            if reward.sum() > 0:
                successes.append(info)
            else:
                failures.append(info)

    # ======================================================
    # create videos
    # ======================================================
    inverse_vocab = {indx: word for word, indx in instr_preprocessor.vocab.vocab.items()}

    print("Successes:",  len(successes))
    print("Failures:",  len(failures))

    if len(failures) == 0 and len(successes) == 0:
        failures.append(dict(
            env_data=samples.env,
            agent_data=samples.agent,
            ))

    for idx, trajectory in enumerate(successes[:num_success]):
        make_video(trajectory,
            config=config,
            video_path=os.path.join(path, f"success_{idx}.mp4"),
            inverse_vocab=inverse_vocab,
            xlim=xlim, ylim=xlim,
            title_size=title_size,
            fps=fps)

    for idx, trajectory in enumerate(failures[:num_failure]):
        make_video(trajectory,
            config=config,
            video_path=os.path.join(path, f"failure{idx}.mp4"),
            inverse_vocab=inverse_vocab,
            xlim=xlim, ylim=xlim,
            title_size=title_size,
            fps=fps)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path',
        help='path to params',
        default=None)
    parser.add_argument('--n_parallel',
        help='number of sampler workers',
        type=int,
        default=1)
    parser.add_argument('--asynch',
        help='whether to run asynchronously',
        type=int,
        default=0)
    parser.add_argument('--batch_T',
        help='number of timesteps for each data collection batch',
        type=int,
        default=2000)
    parser.add_argument('--trajectories',
        help='number of times to collect batches before timing out',
        type=int,
        default=1)
    parser.add_argument('--verbosity',
        help='verbosity',
        type=int,
        default=0)
    parser.add_argument('--num-success',
        help='number of sampler workers',
        type=int,
        default=5)
    parser.add_argument('--num-failure',
        help='number of sampler workers',
        type=int,
        default=5)

    args = parser.parse_args()
    main(**vars(args))
    # main()
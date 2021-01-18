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
from sfgen.babyai.configs import configs
from sfgen.babyai.env import BabyAIEnv
from sfgen.tools.video_maker import image_initialization, update_image, VideoMaker
from experiments.babyai_exp import load_algo_agent, load_instr_preprocessor

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
            print("Found params in path")
            print("="*25)
            return params, settings
        else:
            raise RuntimeError(f"Counldn't find 'params.pkl' in path: {path}")
    elif os.path.isfile(path):
        raise RuntimeError(f"Please provide directory only")
    else:
        raise RuntimeError(f"Neither dir nor file: {path}")


def main(path,
    asynch=False,
    n_parallel=2,
    num_success=10,
    num_failure=10,
    trajectories=100,
    verbosity=0,
    fps=1,
    **kwargs):

    # ======================================================
    # load settings
    # ======================================================
    pkl_file, settings_file = load_filename(path)
    with open(settings_file, 'r') as f:
        config = json.load(f)

    instr_preprocessor=load_instr_preprocessor()
    config['env'].update(
        dict(instr_preprocessor=instr_preprocessor,
            verbosity=verbosity,
            ))

    ckpt = torch.load(pkl_file)

    algo, agent = load_algo_agent(config,
        agent_kwargs=dict(
            initial_model_state_dict=ckpt['agent_state_dict'],
            ))
    
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
        eval_env_kwargs=config['env'],
        batch_T=100,
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
            reward = samples.env.reward[s:e+1]
            done = samples.env.done[s:e+1]
            # length = e - s
            # samples.env.observation.mission[s:e]
            # import ipdb; ipdb.set_trace()
            info = dict(
                # reward=reward,
                env_data=samples.env[s:e+1],
                agent_data=samples.agent[s:e+1],
                indices=(s,e),
                )
            if reward.sum() > 0:
                successes.append(info)
            else:
                failures.append(info)

    # ======================================================
    # create videos
    # ======================================================
    inverse_vocab = {indx: word for word, indx in instr_preprocessor.vocab.vocab.items()}

    def mission2text(mission):
        indices = [int(i.item()) for i in mission]
        words = [inverse_vocab[i] for i in indices]
        return " ".join(words)

    # successes[0]['indices']
    # t0 = mission2text(samples.env.observation.mission[0,0])
    # t1 = mission2text(samples.env.observation.mission[1,0])

    def make_video(trajectory, video_path):
        xlim = 3
        ylim = 3
        obs = trajectory['env_data'].observation

        task = mission2text(obs.mission[0,0])

        length = obs.mission.shape[0]
        reward = trajectory['env_data'].reward



        boxes = {
            'image': {'x':[0,5], 'y':[0,ylim]},
        }
        initialization_fns={
            'image': partial(image_initialization, 
                title=f"{task}\nlength={length}\nreward={reward.sum()}",
                title_size=12),
        }
        update_fns = {
            'image': update_image
        }

        video_maker = VideoMaker(
            xlim=xlim, ylim=ylim, boxes=boxes,
            fps=fps,
            settings=config,
            update_fns=update_fns,
            verbosity=1,
            initialization_fns=initialization_fns,
        )

        video_maker.make(
            env_data=trajectory['env_data'],
            agent_data=trajectory['agent_data'],
            length=length,
            video_path=video_path,
        )
    for idx, trajectory in enumerate(successes[:num_success]):
        make_video(trajectory,
            video_path=os.path.join(path, f"success_{idx}.mp4"))
    for idx, trajectory in enumerate(failures[:num_failure]):
        make_video(trajectory,
            video_path=os.path.join(path, f"failure{idx}.mp4"))

    # import ipdb; ipdb.set_trace()



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
        help='number of sampler workers',
        type=int,
        default=0)
    parser.add_argument('--verbosity',
        help='number of sampler workers',
        type=int,
        default=0)
    parser.add_argument('--trajectories',
        help='number of sampler workers',
        type=int,
        default=50)
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
import ipdb

from sfgen.babyai_kitchen.levelgen import KitchenLevel
from gym_minigrid.wrappers import RGBImgPartialObsWrapper
import gym_minigrid.window

import matplotlib.pyplot as plt

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--level', help='BabyAI level', default='GoToLocal')
    parser.add_argument('--num-missions', help='# of unique missions', default=10)
    parser.add_argument('--num-distractors', type=int, default=0)
    parser.add_argument('--room-size', type=int, default=8)
    parser.add_argument('--agent-view-size', type=int, default=3)
    parser.add_argument('--render-mode', type=str, default='human')
    parser.add_argument('--task-kinds', type=str, default=['cook', 'clean', 'slice'], nargs="+")
    
    parser.add_argument('--actions', type=str, default=['left', 'right', 'forward', 'pickup', 'place', 'toggle', 'slice'], nargs="+")
    parser.add_argument('--objects', type=str, default=[], nargs="+")
    parser.add_argument('--random-object-state', type=int, default=1)
    parser.add_argument('--state-yaml', type=str, default=None)
    parser.add_argument('--num-rows', type=int, default=1)
    parser.add_argument('--tile-size', type=int, default=32)
    parser.add_argument('--steps', type=int, default=1)
    parser.add_argument('--seed', type=int, default=9)
    parser.add_argument('--verbosity', type=int, default=2)
    args = parser.parse_args()

    # env_class = getattr(iclr19_levels, "Level_%s" % args.level)

    kwargs={}

    kwargs['num_dists'] = args.num_distractors
    if args.num_rows:
        kwargs['num_rows'] = args.num_rows
        kwargs['num_cols'] = args.num_rows
    env = KitchenLevel(
        room_size=args.room_size,
        agent_view_size=args.agent_view_size,
        random_object_state=args.random_object_state,
        task_kinds=args.task_kinds,
        actions=args.actions,
        objects=args.objects,
        verbosity=args.verbosity,
        load_actions_from_tasks=True,
        use_time_limit=False,
        seed=args.seed,
        **kwargs)
    # mimic settings during training
    env = RGBImgPartialObsWrapper(env, tile_size=args.tile_size)
    render_kwargs = {'tile_size' : env.tile_size}

    window = gym_minigrid.window.Window('kitchen')
    window.show(block=False)

    def forward():
        x, _, _, _ = env.step(2); 
        env.render()
        window.show_img(x['image'])
    def left():
        x, _, _, _ = env.step(0); 
        env.render()
        window.show_img(x['image'])
    def right():
        x, _, _, _ = env.step(1); 
        env.render()
        window.show_img(x['image'])

    for mission_indx in range(args.num_missions):
        env.seed(mission_indx)
        print("="*50)
        print("Reset")
        print("="*50)
        obs = env.reset()
        print("Task:", obs['mission'])
        topdown = env.render('human', **render_kwargs)
        window.show_img(obs['image'])
        # ipdb.set_trace()
        for step in range(args.steps):
            obs, reward, done, info = env.step(env.action_space.sample())
            topdown = env.render('human', **render_kwargs)
            window.show_img(obs['image'])

            if done:
                print(f"Complete! Reward: {reward}")
                print(f"info: {str(info)}")
                print(f"Episode length: {step+1}")
                break

        ipdb.set_trace()


if __name__ == "__main__":
    main()

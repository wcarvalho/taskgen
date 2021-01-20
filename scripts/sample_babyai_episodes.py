import ipdb
import babyai.levels.iclr19_levels as iclr19_levels

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--level', help='BabyAI level', default='GoToLocal')
    parser.add_argument('--num-missions', help='# of unique missions', default=10)
    parser.add_argument('--num-distractors', type=int, default=0)
    parser.add_argument('--room-size', type=int, default=8)
    parser.add_argument('--num-rows', type=int, default=0)
    parser.add_argument('--steps', type=int, default=1)
    args = parser.parse_args()

    env_class = getattr(iclr19_levels, "Level_%s" % args.level)

    kwargs={}
    if args.num_distractors:
        kwargs['num_dists'] = args.num_distractors

    if args.num_rows:
        kwargs['num_rows'] = args.num_rows
        kwargs['num_cols'] = args.num_rows
    env = env_class(room_size=args.room_size, **kwargs)
    env.render('human')

    for mission_indx in range(args.num_missions):
        env.seed(mission_indx)
        obs = env.reset()
        print(obs['mission'])
        # action=1; env.step(action); env.render()
        for step in range(args.steps):
            obs, _, _, _ = env.step(env.action_space.sample())
            env.render('human')
        ipdb.set_trace()

if __name__ == "__main__":
    main()

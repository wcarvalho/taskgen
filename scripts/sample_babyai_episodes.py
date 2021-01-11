import ipdb
import babyai.levels.iclr19_levels as iclr19_levels

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--level', help='BabyAI level', default='GoToLocal')
    parser.add_argument('--num-missions', help='# of unique missions', default=10)
    parser.add_argument('--num-distractors', type=int, default=2)
    parser.add_argument('--room-size', type=int, default=12)
    args = parser.parse_args()

    env_class = getattr(iclr19_levels, "Level_%s" % args.level)
    env = env_class(room_size=args.room_size, num_dists=args.num_distractors)


    for mission_indx in range(args.num_missions):
        env.seed(mission_indx)
        obs = env.reset()
        print(obs['mission'])
        env.render()
        ipdb.set_trace()

if __name__ == "__main__":
    main()

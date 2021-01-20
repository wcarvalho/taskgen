import ipdb

from sfgen.babyai_kitchen.levelgen import KitchenLevel

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--level', help='BabyAI level', default='GoToLocal')
    parser.add_argument('--num-missions', help='# of unique missions', default=10)
    parser.add_argument('--num-distractors', type=int, default=0)
    parser.add_argument('--room-size', type=int, default=8)
    parser.add_argument('--agent-view-size', type=int, default=3)
    parser.add_argument('--random-object-state', type=int, default=1)
    parser.add_argument('--num-rows', type=int, default=1)
    parser.add_argument('--steps', type=int, default=1)
    parser.add_argument('--verbosity', type=int, default=2)
    args = parser.parse_args()

    # env_class = getattr(iclr19_levels, "Level_%s" % args.level)

    kwargs={}
    # if args.num_distractors:
    #     kwargs['num_dists'] = args.num_distractors

    if args.num_rows:
        kwargs['num_rows'] = args.num_rows
        kwargs['num_cols'] = args.num_rows
    env = KitchenLevel(
        room_size=args.room_size,
        agent_view_size=args.agent_view_size,
        random_object_state=args.random_object_state,
        verbosity=args.verbosity,
        **kwargs)

    def forward(): env.step(2); env.render()
    def left(): env.step(0); env.render()
    def right(): env.step(1); env.render()

    for mission_indx in range(args.num_missions):
        env.seed(mission_indx)
        print("="*50)
        print("Reset")
        print("="*50)
        obs = env.reset()
        print("Task:", obs['mission'])
        # action=1; env.step(action); env.render()
        for step in range(args.steps):
            # obs, _, _, _ = env.step(env.action_space.sample())

            env.render('human')

        ipdb.set_trace()

if __name__ == "__main__":
    main()

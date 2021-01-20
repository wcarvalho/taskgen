import ipdb
from enum import IntEnum
from babyai.levels.levelgen import RoomGridLevel, LevelGen, RejectSampling
from sfgen.babyai_kitchen.world import KitchenObject, Food, Kichenware

class KitchenLevel(RoomGridLevel):
    """
    """
    def __init__(
        self,
        room_size=12,
        num_rows=1,
        num_cols=1,
        num_dists=18,
        # locked_room_prob=0,
        locations=True,
        unblocking=True,
        implicit_unlock=True,
        random_object_state=False,
        action_kinds=['goto', 'pickup', 'open', 'putnext'],
        instr_kinds=['action', 'and', 'seq'],
        seed=None,
        **kwargs,
    ):
        self.num_dists = num_dists
        # self.locked_room_prob = locked_room_prob
        self.locations = locations
        self.unblocking = unblocking
        self.implicit_unlock = implicit_unlock
        self.action_kinds = action_kinds
        self.instr_kinds = instr_kinds
        self.random_object_state = random_object_state

        self.locked_room = None

        self.default_objects = self._default_objects()
        self.object2idx = {obj.name : idx for idx, obj in enumerate(self.default_objects)}

        super().__init__(
            room_size=room_size,
            num_rows=num_rows,
            num_cols=num_cols,
            seed=seed,
            **kwargs,
        )

    @staticmethod
    def _default_objects():
        return [
                KitchenObject("sink", rendering_scale=96),
                KitchenObject("stove", rendering_scale=96),
                KitchenObject("knife", rendering_scale=96),
                Kichenware("pot", rendering_scale=96),
                Kichenware("pan", rendering_scale=96),
                Food('lettuce', cookable=True, sliceable=True),
                Food('potato', cookable=True, sliceable=True),
                Food('tomato', cookable=True, sliceable=True),
                Food('onion', cookable=True, sliceable=True),
                # Food('apple', cookable=False, sliceable=True),
        ]

    # Enumeration of possible actions
    class Actions(IntEnum):
        # Turn left, turn right, move forward
        left = 0
        right = 1
        forward = 2

        # Pick up an object
        pickup = 3

        # Drop an object
        drop = 4

        # Toggle/activate an object
        toggle = 5

        # place: place in front of agent
        place = 6

        # slice: must be holding knife to slice in front of agent
        slice = 7

        # fill: must be in front of sink
        fill = 8 

        # fill: must be in front of sink
        clean = 9

        # Done completing task
        done = 10


    def add_objects(self, i=None, j=None, num_distractors=10, all_unique=True):
        """
        Add random objects that can potentially distract/confuse the agent.
        """

        # # -----------------------
        # # Collect a list of existing objects (always 0?)
        # # -----------------------
        # objs = []
        # for row in self.room_grid:
        #     for room in row:
        #         for obj in room.objs:
        #             objs.append((obj.type, obj.color))


        objs = []
        for obj in self.default_objects:
            self.place_in_room(0, 0, obj)
            if self.random_object_state:
                obj.random_state()
            objs.append(obj)
            obj.set_id(self.object2idx[obj.name])

        return objs

        # # -----------------------
        # # List of distractors added
        # # -----------------------
        # dists = []
        # while len(dists) < num_distractors:
        #     color = self._rand_elem(COLOR_NAMES)
        #     type = self._rand_elem(['key', 'ball', 'box'])
        #     obj = (type, color)

        #     if all_unique and obj in objs:
        #         continue

        #     # Add the object to a random room if no room specified
        #     room_i = i
        #     room_j = j
        #     if room_i == None:
        #         room_i = self._rand_int(0, self.num_cols)
        #     if room_j == None:
        #         room_j = self._rand_int(0, self.num_rows)

        #     dist, pos = self.add_object(room_i, room_j, *obj)

        #     objs.append(obj)
        #     dists.append(dist)

        # return default_objects[:1]

    def _gen_grid(self, **kwargs):
        """dependencies between RoomGridLevel, MiniGridEnv, and RoomGrid are pretty confusing so just call base _gen_grid function to generate grid.
        """
        super(RoomGridLevel, self)._gen_grid(**kwargs)


    def generate_task(self):
        """copied from babyai.levels.levelgen:LevelGen.gen_mission
        """

        # connect all rooms
        self.connect_all()

        self.add_objects(num_distractors=self.num_dists, all_unique=False)

        # The agent must be placed after all the object to respect constraints
        while True:
            self.place_agent()
            start_room = self.room_from_pos(*self.agent_pos)
            # Ensure that we are not placing the agent in the locked room
            if start_room is self.locked_room:
                continue
            break

        # If no unblocking required, make sure all objects are
        # reachable without unblocking
        if not self.unblocking:
            self.check_objs_reachable()

        # Generate random instructions
        return None
        return self.rand_instr(
            action_kinds=self.action_kinds,
            instr_kinds=self.instr_kinds
        )


    def vaidation_task(self, task):
        pass

    def reset_task(self):
        """copied from babyai.levels.levelgen:RoomGridLevel._gen_drid
        - until success:
            - generate grid
            - generate task
                - generate objects
                - place object
                - generate language instruction
            - validate instruction
        """
        # We catch RecursionError to deal with rare cases where
        # rejection sampling gets stuck in an infinite loop
        while True:
            try:
                self._gen_grid(width=self.width, height=self.height)

                # Generate the mission
                task = self.generate_task()

                # Validate the instructions
                self.vaidation_task(task)

            except RecursionError as error:
                print('Timeout during mission generation:', error)
                continue

            except RejectSampling as error:
                #print('Sampling rejected:', error)
                continue

            break

        return task

    def reset(self, **kwargs):
        """Copied from: 
        - gym_minigrid.minigrid:MiniGridEnv.reset
        - babyai.levels.levelgen:RoomGridLevel.reset
        the dependencies between RoomGridLevel, MiniGridEnv, and RoomGrid were pretty confusing so I rewrote the base reset function.
        """
        # ======================================================
        # copied from: gym_minigrid.minigrid:MiniGridEnv.reset
        # ======================================================
        # reset current position and direction of the agent
        self.agent_pos = None
        self.agent_dir = None

        # -----------------------
        # generate:
        # - grid
        # - objects
        # - agent location
        # - instruction
        # -----------------------
        self.task = self.reset_task()
        if self.task is not None:
            self.surface = self.task.surface(self)
            self.mission = self.surface
        else:
            self.surface = self.mission = "No task"

        # These fields should be defined by _gen_grid
        assert self.agent_pos is not None
        assert self.agent_dir is not None

        # Check that the agent doesn't overlap with an object
        start_cell = self.grid.get(*self.agent_pos)
        assert start_cell is None or start_cell.can_overlap()

        # Item picked up, being carried, initially nothing
        self.carrying = None

        # Step count since episode start
        self.step_count = 0

        # Return first observation
        obs = self.gen_obs()

        # ======================================================
        # copied from babyai.levels.levelgen:RoomGridLevel.reset
        # ======================================================
        # Recreate the verifier
        if self.task:
            self.task.reset_verifier(self)

        # Compute the time step limit based on the maze size and instructions
        nav_time_room = self.room_size ** 2
        nav_time_maze = nav_time_room * self.num_rows * self.num_cols
        if self.task:
            num_navs = self.num_navs_needed(self.task)
        else:
            num_navs = 1
        self.max_steps = num_navs * nav_time_maze

        return obs

    def step(self, action):
        """Copied from: 
        - babyai.levels.levelgen:RoomGridLevel.step
        This class derives from RoomGridLevel. We want to use the parent of RoomGridLevel for step. 
        """
        obs, reward, done, info = super(RoomGridLevel, self).step(action)

        # If we drop an object, we need to update its position in the environment
        if action == self.actions.drop:
            import ipdb; ipdb.set_trace()
            # self.update_objs_poss()

        # If we've successfully completed the mission
        if self.task is not None:
            status = self.task.verify(action)

            if status is 'success':
                done = True
                reward = self._reward()
            elif status is 'failure':
                done = True
                reward = 0

        return obs, reward, done, info

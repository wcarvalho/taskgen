import ipdb
import numpy as np
from enum import IntEnum

from gym import spaces

from babyai.levels.levelgen import RoomGridLevel, LevelGen, RejectSampling


from sfgen.babyai_kitchen.world import Kitchen
from sfgen.babyai_kitchen.tasks import KitchenTask, CleanTask, SliceTask, CookTask

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
        actions = ['left', 'right', 'forward', 'pickup', 'place', 'toggle', 'slice'],
        task_kinds=['slice', 'clean', 'cook'],
        instr_kinds=['action'],
        use_subtasks=False,
        seed=None,
        verbosity=0,
        **kwargs,
    ):
        self.num_dists = num_dists
        # self.locked_room_prob = locked_room_prob
        self.locations = locations
        self.unblocking = unblocking
        self.implicit_unlock = implicit_unlock
        if isinstance(task_kinds, list):
            self.task_kinds = task_kinds
        elif isinstance(task_kinds, str):
            self.task_kinds = [task_kinds]
        else:
            RuntimeError(f"Don't know how to read task kind(s): {str(task_kinds)}")
        self.instr_kinds = instr_kinds
        self.random_object_state = random_object_state
        self.use_subtasks = use_subtasks

        self.verbosity = verbosity
        self.locked_room = None

        self.kitchen = Kitchen(verbosity=verbosity)

        super().__init__(
            room_size=room_size,
            num_rows=num_rows,
            num_cols=num_cols,
            seed=seed,
            **kwargs,
        )

        # ======================================================
        # action space
        # ======================================================
        self.actions = {action:idx for idx, action in enumerate(actions, start=0)}
        self.idx2action = {idx:action for idx, action in enumerate(actions, start=0)}
        self.action_names = actions
        self.action_space = spaces.Discrete(len(self.actions))

    # # Enumeration of possible actions

    # class Actions(IntEnum):
    #     # Turn left, turn right, move forward
    #     left = 0
    #     right = 1
    #     forward = 2

    #     # Pick up an object
    #     pickup = 3

    #     # place an object
    #     place = 4

    #     # Toggle/activate an object
    #     toggle = 5


    #     # slice: must be holding knife to slice in front of agent
    #     slice = 6

    #     # Done completing task
    #     done = 7


    # def add_objects(self, i=None, j=None, num_distractors=10, all_unique=True):
    #     """
    #     Add random objects that can potentially distract/confuse the agent.
    #     """

    #     self.kitchen.reset(randomize_states=self.random_object_state)
    #     for obj in self.kitchen.objects:
    #         self.place_in_room(0, 0, obj)

    def _gen_grid(self, *args, **kwargs):
        """dependencies between RoomGridLevel, MiniGridEnv, and RoomGrid are pretty confusing so just call base _gen_grid function to generate grid.
        """
        super(RoomGridLevel, self)._gen_grid(*args, **kwargs)

    def rand_task(
        self,
        task_kinds,
        instr_kinds,
        use_subtasks,
        depth=0
        ):

        instruction_kind = np.random.choice(instr_kinds)

        if instruction_kind == 'action':
            action_kind = np.random.choice(task_kinds)

            if action_kind.lower() == 'cook':
                task = CookTask(env=self.kitchen)
            elif action_kind.lower() == 'clean':
                task = CleanTask(env=self.kitchen)
            elif action_kind.lower() == 'slice':
                task = SliceTask(env=self.kitchen)
            else:
                raise NotImplementedError(f"Task kind '{action_kind}' not supported.")

        else:
            raise RuntimeError(f"Instruction kind not supported: '{instruction_kind}'")

        return task


    def add_objects(self, task=None, num_distractors=10):
        """
        - if have task, place task objects
        
        Args:
            task (None, optional): Description
            num_distactors (int, optional): Description
        """
        placed_objects = set()

        # first place task objects
        if task is not None:
            for obj in task.task_objects:
                self.place_in_room(0, 0, obj)
                placed_objects.add(obj.type)
                if self.verbosity > 1:
                    print(f"Added task object: {obj.type}")

        # if number of left over objects is less than num_distractors, set as that
        # possible_space = (self.grid.width - 2)*(self.grid.height - 2)
        num_leftover_objects =len(self.kitchen.objects)-len(placed_objects)
        num_distractors = min(num_leftover_objects, num_distractors)


        distractors_added = []
        num_tries = 0
        while len(distractors_added) < num_distractors:
            # infinite loop catch
            num_tries += 1
            if num_tries > 1000:
                raise RuntimeError("infinite loop in `add_objects`")

            # sample objects
            random_object = np.random.choice(self.kitchen.objects)

            # if already added, try again
            if random_object.type in placed_objects:
                continue

            self.place_in_room(0, 0, random_object)
            distractors_added.append(random_object.type)
            placed_objects.add(random_object.type)
            if self.verbosity > 1:
                print(f"Added distractor: {random_object.type}")

    def generate_task(self):
        """copied from babyai.levels.levelgen:LevelGen.gen_mission
        """

        # connect all rooms
        self.connect_all()

        # reset kitchen objects
        self.kitchen.reset(randomize_states=self.random_object_state)

        # Generate random instructions
        task = self.rand_task(
            task_kinds=self.task_kinds,
            instr_kinds=self.instr_kinds,
            use_subtasks=self.use_subtasks,
        )

        self.add_objects(task=task, num_distractors=self.num_dists)


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


        return task


    def vaidate_task(self, task):
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
        tries = 0
        while True:
            tries += 1
            if tries > 1000:
                raise RuntimeError("can't sample task???")
            try:
                # generate grid of observation
                self._gen_grid(width=self.width, height=self.height)

                # Generate the task
                task = self.generate_task()

                # Validate the task
                self.vaidate_task(task)


            except RecursionError as error:
                print(f'Timeout during mission generation:{tries}/100\n', error)
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
            num_navs = self.task.num_navs
        else:
            num_navs = 1
        self.max_steps = num_navs * nav_time_maze

        return obs

    def interact(self, action, object_infront, fwd_pos):
        # Pick up an object
        if action == self.actions.get('pickup', -1):
            if object_infront and object_infront.can_pickup():
                if self.carrying is None:
                    self.carrying = object_infront
                    self.carrying.cur_pos = np.array([-1, -1])
                    self.grid.set(*fwd_pos, None)

        # place an object in front
        elif action == self.actions.get('place', -1):
            if self.carrying and object_infront is not None:
                if hasattr(object_infront, "kitchen_object"):
                    self.carrying = self.kitchen.place(self.carrying, object_infront)
                else:
                    if not object_infront:
                        self.grid.set(*fwd_pos, self.carrying)
                        self.carrying.cur_pos = fwd_pos
                        self.carrying = None

                    # place inside object in front if possible?
                    # custom place command



        # Toggle/activate an object
        elif action == self.actions.get('toggle', -1):
            if object_infront:
                if hasattr(object_infront, 'kitchen_object'):
                    object_infront.toggle(self.kitchen, fwd_pos)
                else:
                    # backwards compatibility
                    object_infront.toggle(self, fwd_pos)


        # slice
        elif action == self.actions.get('slice', -1):
            if object_infront and self.carrying:
                if hasattr(object_infront, 'kitchen_object'):
                    object_infront.slice(self.kitchen, fwd_pos, self.carrying)

                else:
                    # not supported, nothing happens
                    pass


        else:
            raise RuntimeError(f"Unknown action: {action}")


    def step(self, action):
        """Copied from: 
        - gym_minigrid.minigrid:MiniGridEnv.step
        - babyai.levels.levelgen:RoomGridLevel.step
        This class derives from RoomGridLevel. We want to use the parent of RoomGridLevel for step. 
        """
        # ======================================================
        # copied from MiniGridEnv
        # ======================================================
        self.step_count += 1

        reward = 0
        done = False

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        object_infront = self.grid.get(*fwd_pos)

        print(self.idx2action[action], object_infront)
        # Rotate left
        if action == self.actions.get('left', -1):
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4

        # Rotate right
        elif action == self.actions.get('right', -1):
            self.agent_dir = (self.agent_dir + 1) % 4

        # Move forward
        elif action == self.actions.get('forward', -1):
            if object_infront == None or object_infront.can_overlap():
                self.agent_pos = fwd_pos
            if object_infront != None and object_infront.type == 'goal':
                done = True
                reward = self._reward()
            if object_infront != None and object_infront.type == 'lava':
                done = True
        else:
            self.interact(action, object_infront, fwd_pos)


        if self.step_count >= self.max_steps:
            done = True

        # ======================================================
        # copied from RoomGridLevel
        # ======================================================

        # If we drop an object, we need to update its position in the environment
        # if action == self.actions.get('drop', -1):
        #     import ipdb; ipdb.set_trace()
            # self.update_objs_poss()

        # If we've successfully completed the mission
        if self.task is not None:
            give_reward, done = self.task.check_status()

            if give_reward:
                reward = self._reward()
            else:
                reward = 0

        obs = self.gen_obs()
        info = {}
        return obs, reward, done, info

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
        objects = [],
        actions = ['left', 'right', 'forward', 'pickup_container', 'pickup_contents', 'place', 'toggle', 'slice'],
        task_kinds=['slice', 'clean', 'cook'],
        instr_kinds=['action'],
        use_subtasks=False,
        use_time_limit=True,
        seed=None,
        verbosity=0,
        **kwargs,
    ):
        self.num_dists = num_dists
        # self.locked_room_prob = locked_room_prob
        self.use_time_limit = use_time_limit
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

        # define the dynamics of the objects with kitchen
        self.kitchen = Kitchen(objects=objects, verbosity=verbosity)
        self.check_task_actions = False

        # to avoid checking task during reset of initialization
        super().__init__(
            room_size=room_size,
            num_rows=num_rows,
            num_cols=num_cols,
            seed=seed,
            **kwargs,
        )
        self.check_task_actions = True

        # ======================================================
        # action space
        # ======================================================
        self.actions = {action:idx for idx, action in enumerate(actions, start=0)}
        self.idx2action = {idx:action for idx, action in enumerate(actions, start=0)}
        self.action_names = actions
        self.action_space = spaces.Discrete(len(self.actions))


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
            elif action_kind.lower() == 'none':
                task = None
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
        num_leftover_objects = len(self.kitchen.objects)-len(placed_objects)
        num_distractors = min(num_leftover_objects, num_distractors)

        if len(placed_objects) == 0:
            num_distractors = max(num_distractors, 1)

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
        if task is not None and self.check_task_actions:
            task.check_actions(self.action_names)


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
        # when call reset during initialization, don't load
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

        # updating carrying in kitchen env just in case
        self.kitchen.update_carrying(self.carrying)
        # ======================================================
        # copied from babyai.levels.levelgen:RoomGridLevel.reset
        # ======================================================
        # # Recreate the verifier
        # if self.task:
        #     import ipdb; ipdb.set_trace()
        #     self.task.reset_verifier(self)

        # Compute the time step limit based on the maze size and instructions
        nav_time_room = int(self.room_size ** 2.5)
        nav_time_maze = nav_time_room * self.num_rows * self.num_cols
        if self.task:
            num_navs = self.task.num_navs
        else:
            num_navs = 1
        self.max_steps = num_navs * nav_time_maze

        return obs


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


        # Rotate left
        action_info = None
        interaction = False
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
            action_info = self.kitchen.interact(
                action=self.idx2action[action],
                object_infront=object_infront,
                fwd_pos=fwd_pos,
                grid=self.grid,
                env=self, # only used for backwards compatibility with toggle
            )
            self.carrying = self.kitchen.carrying
            interaction = True

        step_info = self.kitchen.step()

        if self.verbosity > 1:
            from pprint import pprint
            print('='*50)
            obj_type = object_infront.type if object_infront else None
            print(self.idx2action[action], obj_type)
            pprint(action_info)
            print('-'*10, 'Env Info', '-'*10)
            print("Carrying:", self.carrying)
            if self.task is not None:
                print(f"task objects:")
                pprint(self.task.task_objects)
            else:
                print(f"env objects:")
                pprint(self.kitchen.objects)

            # if isinstance(action_info, list):
            #     success = sum([a['success'] for a in action_info]) > 0
            # elif isinstance(action_info, dict):
            #     success = action_info['success']
            # elif action_info is None:
            #     success = None
            # else:
            #     raise RuntimeError
            # if success:
            #     # self.grid.get(*fwd_pos)
            #     self.render()
            #     import ipdb; ipdb.set_trace()

        # if action_success and interaction:
        #     import ipdb; ipdb.set_trace()

        # ======================================================
        # copied from RoomGridLevel
        # ======================================================
        # If we drop an object, we need to update its position in the environment
        # if action == self.actions.get('drop', -1):
        #     import ipdb; ipdb.set_trace()
            # self.update_objs_poss()

        # If we've successfully completed the mission
        info = {'success': False}
        if self.task is not None:
            give_reward, done = self.task.check_status()

            if done:
                info['success'] = True
            if give_reward:
                reward = self._reward()
            else:
                reward = 0

        # if past step count, done
        if self.step_count >= self.max_steps and self.use_time_limit:
            import ipdb; ipdb.set_trace()
            done = True

        obs = self.gen_obs()
        return obs, reward, done, info

    def _reward(self):
        """
        Compute the reward to be given upon success
        """

        return 1

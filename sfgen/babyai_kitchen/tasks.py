import numpy as np
from babyai.levels.verifier import Instr

class KitchenTask(Instr):
    """docstring for KitchenTasks"""
    def __init__(self, env):
        super(KitchenTask, self).__init__()
        self._task_objects = []
        self.env = env
        self.instruction = self.generate()


    def generate(self):
        raise NotImplemented

    @property
    def task_objects(self):
        return self._task_objects

    def surface(self, *args, **kwargs):
        return self.instruction

    @property
    def num_navs(self):
        return 1

    def __repr__(self):
        string = self.instruction
        if self.task_objects:
            for object in self.task_objects:
                string += "\n" + str(object)

        return string

    def check_status(self):
        return False, False

    def check_actions(self, actions):
        for action in self.task_actions():
            if action == 'pickup':
                assert 'pickup_contents' in actions or 'pickup_container' in actions
            elif action == 'pickup_and':
                assert 'pickup_contents' in actions and 'pickup_container' in actions
            else:
                assert action in actions

    @staticmethod
    def task_actions():
        return [
            'toggle',
            'pickup_and',
            'place'
            ]

class CleanTask(KitchenTask):
    """docstring for CleanTask"""

    def generate(self):
        objects_to_clean = self.env.objects_with_property(['dirty'])
        self.object_to_clean = np.random.choice(objects_to_clean)
        self.object_to_clean.set_prop('dirty', True)


        self.sink = self.env.objects_by_type(["sink"])[0]
        self.sink.set_prop('on', False)

        self._task_objects = [self.object_to_clean, self.sink]
        return f"clean {self.object_to_clean.name}"

    @property
    def num_navs(self): return 1

    def check_status(self):
        done = reward = self.object_to_clean.state['dirty'] == False

        return reward, done

class SliceTask(KitchenTask):
    """docstring for SliceTask"""

    def generate(self):
        objects_to_slice = self.env.objects_with_property(['sliced'])
        self.object_to_slice = np.random.choice(objects_to_slice)
        self.object_to_slice.set_prop('sliced', False)

        self.knife = self.env.objects_by_type(["knife"])[0]

        self._task_objects = [self.object_to_slice, self.knife]
        return f"slice {self.object_to_slice.name}"

    @property
    def num_navs(self): return 1

    def check_status(self):
        done = reward = self.object_to_slice.state['sliced'] == True

        return reward, done

    @staticmethod
    def task_actions():
        return [
            'slice',
            'pickup_and',
            'place'
            ]


class CoolTask(KitchenTask):
    """docstring for CookTask"""

    def generate(self):
        self.fridge = self.env.objects_by_type(['fridge'])[0]
        objects_to_cool = self.env.objects_by_type(self.fridge.can_contain)
        
        self.object_to_cool = np.random.choice(objects_to_cool)


        self.object_to_cool.set_prop("temp", "room")
        self.fridge.set_prop("temp", 'room')
        self.fridge.set_prop("on", False)


        self._task_objects = [
            self.object_to_cool,
            self.fridge,
        ]
        return f"cool {self.object_to_cool.name}"

    @property
    def num_navs(self): return 1

    def check_status(self):
        done = reward = self.object_to_cool.state['temp'] == 'cold'

        return reward, done

class HeatTask(KitchenTask):
    """docstring for CookTask"""
    def __init__(self,
        env,
        types_to_heat=[],
        ):
        self._task_objects = []
        self.types_to_heat = types_to_heat
        self.env = env
        self.instruction = self.generate()


    def generate(self):
        self.stove = self.env.objects_by_type(['stove'])[0]
        if self.types_to_heat:
            objects_to_heat = self.env.objects_by_type(self.types_to_heat)
        else:
            objects_to_heat = self.env.objects_by_type(self.stove.can_contain)
        
        self.object_to_heat = np.random.choice(objects_to_heat)


        self.object_to_heat.set_prop("temp", "room")
        self.stove.set_prop("temp", 'room')
        self.stove.set_prop("on", False)


        self._task_objects = [
            self.object_to_heat,
            self.stove,
        ]
        return f"heat {self.object_to_heat.name}"

    @property
    def num_navs(self): return 1

    def check_status(self):
        done = reward = self.object_to_heat.state['temp'] == 'hot'

        return reward, done


class PlaceTask(KitchenTask):
    def __init__(self,
        env,
        container_types=[],
        place_types=[],
        ):
        self.container_types = container_types
        self.place_types = place_types
        self._task_objects = []
        self.env = env
        self.instruction = self.generate()



    def generate(self):
        # which container
        if self.container_types:
            containers = self.env.objects_by_type(self.container_types)
        else:
            containers = [o for o in self.env.objects if o.is_container]
        self.container = np.random.choice(containers)


        if self.place_types:
            choices = self.env.objects_by_type(self.place_types)
        else:
            choices = self.env.objects_by_type(self.container.can_contain)
        # what to place inside container
        self.to_place = np.random.choice(choices)

        self._task_objects = [
            self.container, 
            self.to_place
        ]

        return f"place {self.to_place.type} in {self.container.type}"

    def check_status(self):
        if self.container.contains:
            # let's any match fit, not just the example used for defining the task. 
            # e.g., if multiple pots, any pot will work inside container
            done = reward = self.container.contains.type == self.to_place.type
        else:
            done = reward = False

        return reward, done

    @staticmethod
    def task_actions():
        return [
            'pickup_and',
            'place'
            ]

class CookTask(KitchenTask):
    """docstring for CookTask"""

    def generate(self):
        objects_to_cook = self.env.objects_with_property(['cooked'])
        objects_to_cook_with = self.env.objects_by_type(['pot', 'pan'])


        self.object_to_cook_on = self.env.objects_by_type(['stove'])[0]
        self.object_to_cook = np.random.choice(objects_to_cook)
        self.object_to_cook_with = np.random.choice(objects_to_cook_with)

        self.object_to_cook.set_prop("cooked", False)
        self.object_to_cook.set_prop("temp", 'room')
        self.object_to_cook_with.set_prop("dirty", False)
        self.object_to_cook_on.set_prop("on", False)


        self._task_objects = [
            self.object_to_cook,
            self.object_to_cook_with,
            self.object_to_cook_on
        ]
        return f"cook {self.object_to_cook.name} with {self.object_to_cook_with.name}"

    @property
    def num_navs(self): return 2

    def check_status(self):
        done = reward = self.object_to_cook.state['cooked'] == True

        return reward, done


class CookseqTask(CookTask):

    def generate(self):
        # generate random task using regular cook task
        self.full_task = CookTask(env=self.env)


        container = self.full_task.object_to_cook_with.type
        food = self.full_task.object_to_cook.type
        self.place_task = PlaceTask(
            env=self.env,
            container_types=[container],
            place_types=[food]
            )
        self.heat_task = HeatTask(
            env=self.env,
            types_to_heat=[container]
            )

        self.tasks = [self.place_task, self.heat_task]
        self.idx = 0


        self._task_objects = self.full_task._task_objects
        return f"{self.place_task.instruction}, {self.heat_task.instruction}"

    def surface(self, *args, **kwargs):
        return self.current_task.instruction

    @property
    def on_final_task(self):
        return self.idx == (len(self.tasks) - 1)

    @property
    def current_task(self):
        return self.tasks[self.idx]


    def check_status(self):
        done = False
        rewards = 0
        reward, subtask_done = self.current_task.check_status()

        if self.on_final_task and subtask_done:
            done = True
        else:
            done = False

        if subtask_done:
            self.idx += 1
            num_tasks = len(self.tasks)
            reward = self.idx/num_tasks
        else:
            reward = 0
        return reward, done


# ======================================================
# Composite tasks
# ======================================================
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
            return string + "\n" + str(self.task_objects)
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
        done = give_reward = self.object_to_clean.state['dirty'] == False

        return give_reward, done

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
        done = give_reward = self.object_to_slice.state['sliced'] == True

        return give_reward, done

    @staticmethod
    def task_actions():
        return [
            'slice',
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
        done = give_reward = self.object_to_cook.state['cooked'] == True

        return give_reward, done

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
        done = give_reward = self.object_to_cool.state['temp'] == 'cold'

        return give_reward, done

class HeatTask(KitchenTask):
    """docstring for CookTask"""

    def generate(self):
        self.stove = self.env.objects_by_type(['stove'])[0]
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
        done = give_reward = self.object_to_heat.state['temp'] == 'hot'

        return give_reward, done


class PlaceTask(KitchenTask):

    def generate(self):
        # which container
        containers = [o for o in self.env.objects if o.is_container]
        self.container = np.random.choice(containers)


        # what to place inside container
        choices = self.env.objects_by_type(self.container.can_contain)
        self.to_place = np.random.choice(choices)

        self._task_objects = [
            self.container, 
            self.to_place
        ]

    def check_status(self):
        if self.container.contains:
            # let's any match fit, not just the example used for defining the task. 
            # e.g., if multiple pots, any one of them will work
            done = give_reward = self.container.contains.type == self.to_place.type
        else:
            done = give_reward = False

        return give_reward, done

    @staticmethod
    def task_actions():
        return [
            'pickup_and',
            'place'
            ]

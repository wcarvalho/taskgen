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
        raise NotImplementedError

    def __repr__(self):
        return self.instruction

    def check_status(self):
        return ''

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
        give_reward = self.object_to_slice.state['sliced'] == True
        done = give_reward

        return give_reward, done


class CookTask(KitchenTask):
    """docstring for CookTask"""

    def generate(self):
        objects_to_cook = self.env.objects_with_property(['cooked'])
        objects_to_cook_with = self.env.objects_by_type(['pot', 'pan'])


        self.object_to_cook_on = self.env.objects_by_type(['stove'])[0]
        self.object_to_cook = np.random.choice(objects_to_cook)
        self.object_to_cook_with = np.random.choice(objects_to_cook_with)

        self.object_to_cook.set_prop("cooked", False)
        self.object_to_cook_with.set_prop("dirty", False)


        self._task_objects = [
            self.object_to_cook,
            self.object_to_cook_with,
            self.object_to_cook_on
        ]
        return f"cook {self.object_to_cook.name} with {self.object_to_cook_with.name}"

    @property
    def num_navs(self): return 2

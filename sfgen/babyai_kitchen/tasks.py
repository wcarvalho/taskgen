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
        string = self.instruction
        if self.task_objects:
            return string + "\n" + str(self.task_objects)
        return string

    def check_status(self):
        return False, False

    def check_actions(self, actions):
        pass

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

    def check_actions(self, actions):
        assert "toggle" in actions
        assert "pickup_contents" in actions
        assert "place" in actions

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

    def check_actions(self, actions):
        assert "slice" in actions
        assert "pickup_contents" in actions or "pickup_container" in actions
        assert "place" in actions


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

    def check_actions(self, actions):
        assert "toggle" in actions
        assert "pickup_contents" in actions and "pickup_container" in actions
        assert "place" in actions

from collections import namedtuple

import numpy as np
from PIL import Image

from gym_minigrid.minigrid import WorldObj

FoodState = namedtuple('FoodState', ['cooked', 'sliced'])
KitchenwareState = namedtuple('KitchenwareState', ['dirty'])

ICONPATH='sfgen/babyai_kitchen/icons'

def open_image(image, rendering_scale):
    # image = imread(image)
    image = Image.open(image)
    # return transform.resize(image, (self.rendering_scale,self.rendering_scale), mode='symmetric', preserve_range=True)
    arr = np.array(image.resize((rendering_scale, rendering_scale)))
    # arr[arr[...,-1]==0] = [255,255,255,0]
    return arr

class KitchenObject(WorldObj):
    """docstring for KitchenObject"""
    def __init__(self, name, image_paths=None, rendering_scale=96):
        # super(KitchenObject, self).__init__()
        self.name = self.type = name
        if image_paths:
            self.image_paths = image_paths
        else:
            self.image_paths = {'default' : f"{ICONPATH}/{name}.png"}
        self.states = list(self.image_paths.keys())
        self.state = self.states[0]

        self.rendering_scale = rendering_scale

        self.images = {k : open_image(v, rendering_scale) for k, v in self.image_paths.items()}

        self.init_pos = None
        self.cur_pos = None
        self.object_id = None

    def render(self, screen):
        # pos = self.cur_pos
        # scale = self.rendering_scale
        obj_img = self.state_image()
        np.copyto(screen, obj_img[:, :, :3])

    def random_state(self):
        idx = np.random.randint(len(self.states))
        self.state = self.states[idx]

    def state_image(self): return self.images[self.state]

    def set_id(self, oid): self.object_id = oid

    def state_id(self): return 0

    def encode(self):
        """Encode the a description of this object as a 3-tuple of integers"""

        # State, 0: open, 1: closed, 2: locked
        # if self.is_open:
        #     state = 0
        # elif self.is_locked:
        #     state = 2
        # elif not self.is_open:
        #     state = 1

        return (self.object_id, 0, self.state_id())


class Kichenware(KitchenObject):
    """docstring for Kichenware"""
    def __init__(self, name, dirtyable=True, **kwargs):
        super(Kichenware, self).__init__(name=name,
            image_paths={
            KitchenwareState(True) : f"{ICONPATH}/{name}_dirty.png",
            KitchenwareState(False): f"{ICONPATH}/{name}.png"
            },
            **kwargs
        )
        self.state = KitchenwareState(False)
        self.state2idx = {
            KitchenwareState(False) : 0,
            KitchenwareState(True) : 1
        }

    def state_id(self):
        return self.state2idx[self.state]

    def can_pickup(self):
        """Can the agent pick this up?"""
        return True

    def can_contain(self):
        """Can this contain another object?"""
        return True

    def toggle(self, env, pos):
        """Method to trigger/toggle an action this object performs"""
        import ipdb; ipdb.set_trace()


class Food(KitchenObject):
    """docstring for Food"""
    def __init__(self, name, cookable=True, sliceable=True, **kwargs):
        paths = {}
        self.state2idx = {}
        for cooked in [True, False]:
            for sliced in [True, False]:
                path = f"{name}"
                if sliced:
                    path += "_sliced"
                if cooked:
                    path += "_cooked"
                state = FoodState(cooked, sliced)
                self.state2idx[state] = len(self.state2idx)

                paths[state] = f"{ICONPATH}/{path}.png"

        super(Food, self).__init__(name=name, image_paths=paths, **kwargs)
        self.state = FoodState(False, False)

    def can_pickup(self):
        """Can the agent pick this up?"""
        return True

if __name__ == '__main__':
    Food("knife")
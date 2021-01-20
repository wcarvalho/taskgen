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
    def __init__(self,
            name,
            image_paths=None,
            state=None,
            state2idx={},
            default_state_id=0,
            rendering_scale=96,
            verbosity=0,
            ):
        # super(KitchenObject, self).__init__()
        self.verbosity = verbosity
        # ======================================================
        # load image paths + rendering
        # ======================================================
        if image_paths:
            self.image_paths = image_paths
        else:
            self.image_paths = {'default' : f"{ICONPATH}/{name}.png"}
        self.rendering_scale = rendering_scale
        self.images = {k : open_image(v, rendering_scale) for k, v in self.image_paths.items()}

        # ======================================================
        # load state info
        # ======================================================
        self.name = self.type = name
        self.states = list(self.image_paths.keys())
        self.state = state or self.states[0]
        self.state2idx = state2idx or {'default':0}
        self.default_state_id = default_state_id

        # ======================================================
        # reset position info
        # ======================================================
        self.init_pos = None
        self.cur_pos = None
        self.object_id = None

    def render(self, screen):
        # pos = self.cur_pos
        # scale = self.rendering_scale
        obj_img = self.state_image()
        np.copyto(screen, obj_img[:, :, :3])

    def reset_state(self, random=False):
        # import ipdb; ipdb.set_trace()
        if random:
            idx = np.random.randint(len(self.states))
        else:
            idx = self.default_state_id
        self.state = self.states[idx]
        if self.verbosity > 1:
            print(f'{self.name} resetting to: {idx}/{len(self.states)} = {self.state}')

    def state_image(self):
        if self.verbosity > 1:
            print(f'objects state {self.name}: {self.state}')
        return self.images[self.state]

    def set_id(self, oid): self.object_id = oid

    def state_id(self):
        return self.state2idx[self.state]

    def encode(self):
        """Encode the a description of this object as a 3-tuple of integers"""


        return (self.object_id, 0, self.state_id())

    def set_verbosity(self, v): self.verbosity = v

class Kichenware(KitchenObject):
    """docstring for Kichenware"""
    def __init__(self, name, dirtyable=True, **kwargs):
        super(Kichenware, self).__init__(name=name,
            image_paths={
                KitchenwareState(True) : f"{ICONPATH}/{name}_dirty.png",
                KitchenwareState(False): f"{ICONPATH}/{name}.png"
                },
            state = KitchenwareState(False),
            state2idx = {
                KitchenwareState(False) : 0,
                KitchenwareState(True) : 1
            }
            **kwargs
        )



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
        state2idx = {}
        for cooked in [True, False]:
            for sliced in [True, False]:
                path = f"{name}"
                if sliced:
                    path += "_sliced"
                if cooked:
                    path += "_cooked"
                state = FoodState(cooked, sliced)
                state2idx[state] = len(state2idx)

                paths[state] = f"{ICONPATH}/{path}.png"

        super(Food, self).__init__(
            name=name,
            image_paths=paths,
            state2idx=state2idx,
            state=FoodState(False, False),
             **kwargs)

    def can_pickup(self):
        """Can the agent pick this up?"""
        return True


class Kitchen:
    """docstring for Kitchen"""
    def __init__(self, verbosity=0):
        super(Kitchen, self).__init__()

        self.verbosity = verbosity
        self._objects = self._default_objects()

        self.object2idx = {}
        self.name2object = {}
        for idx, object in enumerate(self._objects):
            object.set_verbosity(self.verbosity)
            # set id
            self.object2idx[object.name] = idx 
            object.set_id(idx)

            self.name2object[object.name] = object



    @property
    def objects(self):
        return self._objects

    def reset(self, randomize_states=False):
        for object in self.objects:
            object.reset_state(random=randomize_states)

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
        

if __name__ == '__main__':
    Food("knife")
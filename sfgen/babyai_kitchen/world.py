from collections import namedtuple
from sklearn.model_selection import ParameterGrid
import numpy as np
from PIL import Image

from gym_minigrid.minigrid import WorldObj
from gym_minigrid.rendering import fill_coords, point_in_rect

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
            # image_paths=None,
            pickupable=True,
            container=False,
            can_contain=[],
            rendering_scale=96,
            verbosity=0,
            # state=None,
            # state2idx={},
            default_state=None,
            properties=[],

            ):
        """Load:
        - all possible object-states
        - image paths for object-states
        
        Args:
            name (TYPE): Description
            image_paths (None, optional): Description
            state (None, optional): Description
            state2idx (dict, optional): Description
            default_state_id (int, optional): Description
            rendering_scale (int, optional): Description
            verbosity (int, optional): Description
            properties (list, optional): Description
        """
        # super(KitchenObject, self).__init__()
        # ======================================================
        # load basics
        # ======================================================
        self.name = self.type = name
        self.pickupable = pickupable
        self.container = container
        self.contains = None
        self.can_contain = []
        self.verbosity = verbosity
        self.rendering_scale = rendering_scale
        self.properties = properties

        # ======================================================
        # load possible object-states & images
        # ======================================================
        if properties:
            states = []
            state2idx = {}
            idx2state = {}
            image_paths = {}
            possible_states = {}
            for prop in properties:
                possible_states[prop]=[True, False]
            possible_states = [i for i in ParameterGrid(possible_states)]
            # -----------------------
            # load: paths, states
            # -----------------------
            for state in possible_states:
                # ensures that always matches ordering of list
                state = {p:state[p] for p in properties}
                key = str(state)
                states.append(key)

                # indx each state
                state2idx[key] = len(state2idx)
                idx2state[state2idx[key]] = state

                # get path for each state
                path = f"{name}"
                for prop in properties:
                    if state[prop]:
                        path += f"_{prop}"
                image_paths[key] = f"{ICONPATH}/{path}.png"
        else:
            image_paths = {'default':  f"{ICONPATH}/{name}.png"}
            state2idx = {'default':  0}
            states = ['default']
            idx2state = {0 : 'default'}


        self.image_paths = image_paths
        self.images = {k : open_image(v, rendering_scale) for k, v in image_paths.items()}

        # ======================================================
        # load state info
        # ======================================================

        self.idx2state = idx2state
        self.state2idx = state2idx
        self.states = states
        if default_state:
            self.state = self.default_state = default_state
        else:
            if properties:
                all_false = {prop: False for prop in properties}
                self.state = self.default_state = all_false
            else:
                self.state = self.default_state = "default"

        self.default_state_id = self.state2idx[str(self.default_state)]


        # ======================================================
        # reset position info
        # ======================================================
        self.init_pos = None
        self.cur_pos = None
        self.object_id = None
        self.kitchen_object = True

    def has_prop(self, prop):
        return prop in self.properties

    def set_prop(self, prop, val):
        self.state[prop] = val

    def render(self, screen):
        obj_img = self.state_image()
        np.copyto(screen, obj_img[:, :, :3])
        fill_coords(screen, point_in_rect(0, 0.031, 0, 1), (100, 100, 100))
        fill_coords(screen, point_in_rect(0, 1, 0, 0.031), (100, 100, 100))

    def reset_state(self, random=False):
        # import ipdb; ipdb.set_trace()
        if random:
            idx = np.random.randint(len(self.states))
        else:
            idx = self.default_state_id
        self.state = self.idx2state[idx]
        if self.verbosity > 1:
            print(f'{self.name} resetting to: {idx}/{len(self.states)} = {self.state}')

    def state_image(self):
        if self.verbosity > 1:
            print(f'objects state {self.name}: {self.state}')
        return self.images[str(self.state)]

    def set_id(self, oid): self.object_id = oid

    def state_id(self):
        return self.state2idx[str(self.state)]

    def encode(self):
        """Encode the a description of this object as a 3-tuple of integers"""

        return (self.object_id, 0, self.state_id())

    def set_verbosity(self, v): self.verbosity = v

    def __repr__(self):
        return f"{self.name}: {str(self.state)}"

    # ======================================================
    # Actions
    # ======================================================
    def can_pickup(self): return self.pickupable
    def accepts(self, object):
        return object.type in self.can_contain

    def slice(self, env, pos, carrying):
        can_slice = self.has_prop('sliced')
        if not can_slice: return

        if carrying.type == 'knife':
            self.set_prop("sliced", True)


    def toggle(self, env, pos):
        can_toggle = self.has_prop('on')
        if not can_toggle: return

        import ipdb; ipdb.set_trace()

    def place(self, env, pos):
        import ipdb; ipdb.set_trace()



class Food(KitchenObject):
    """docstring for Food"""
    def __init__(self,
        name,
        properties=['sliced', 'cooked'],
        default_state={'sliced': False, 'cooked': False},
        **kwargs):
        super(Food, self).__init__(
            name=name,
            properties=properties,
            default_state=default_state,
             **kwargs)


class KitchenContainer(KitchenObject):
    """docstring for KitchenContainer"""
    def __init__(self, hides_content=False, *args, **kwargs):
        super(KitchenContainer, self).__init__(*args, 
            container=True,
            **kwargs)
        self.hides_content = hides_content
        


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

    def place(self, carrying, inside):
        if not inside.container: return carrying

        import ipdb; ipdb.set_trace()

    @property
    def objects(self):
        return self._objects

    def reset(self, randomize_states=False):
        for object in self.objects:
            object.reset_state(random=randomize_states)

    @staticmethod
    def _default_objects():
        return [
                KitchenContainer(
                    name="sink", 
                    properties=['on'],
                    can_contain=['knife', 'pot', 'pan', 'fork', 'plates'],
                    pickupable=False,
                ),
                KitchenContainer(
                    name="stove", 
                    properties=['on'],
                    can_contain=['pot', 'pan'],
                    pickupable=False,
                ),

                KitchenContainer(
                    name="pot", 
                    hides_content=True,
                    can_contain=['lettuce', 'potato', 'tomato', 'onion'],
                    properties=['dirty']
                ),
                KitchenContainer(
                    name="pan",
                    can_contain=['lettuce', 'potato', 'tomato', 'onion'],
                    hides_content=True,
                    properties=['dirty']
                ),
                KitchenContainer(
                    name="plates",
                    can_contain=['lettuce', 'potato', 'tomato', 'onion'],
                    properties=['dirty']
                ),

                KitchenObject(
                    name="fork", 
                    properties=['dirty']
                ),
                KitchenObject(
                    name="knife"
                ),

                Food(name='lettuce'),
                Food(name='potato'),
                Food(name='tomato'),
                Food(name='onion'),

        ]

    def objects_with_property(self, props):
        return [object for object in self.objects 
            if sum([object.has_prop(p) for p in props]) == len(props)
        ]

    def objects_by_type(self, types):
        matches = []
        if isinstance(types, list):
            pass
        elif isinstance(types, str):
            types = [types]
        else:
            raise RuntimeError
        for t in types:
            matches.extend([object for object in self.objects if object.type == t])
        return matches

if __name__ == '__main__':
    Food("knife")
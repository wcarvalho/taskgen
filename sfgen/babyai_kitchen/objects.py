import copy
from collections import namedtuple
from sklearn.model_selection import ParameterGrid
import numpy as np
from PIL import Image

from gym_minigrid.minigrid import WorldObj
from gym_minigrid.rendering import fill_coords, point_in_rect


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
            is_container=False,
            can_heat_contained=False,
            toggle_heats=False,
            can_clean_contained=False,
            can_heat=False,
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
        self.is_container = is_container
        self.can_heat = can_heat
        self.can_heat_contained = can_heat_contained
        self.toggle_heats = toggle_heats
        self.can_clean_contained = can_clean_contained
        assert not (can_heat_contained and can_clean_contained), "can't both heat and clean contents in this dumb grid world"
        self.contains = None
        self.hot = False
        self.can_contain = can_contain
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
            print(f'object image: {self.name}, {self.state}')
        return self.images[str(self.state)]

    def set_id(self, oid): self.object_id = oid

    def state_id(self):
        return self.state2idx[str(self.state)]

    def encode(self):
        """Encode the a description of this object as a 3-tuple of integers"""

        return (self.object_id, 0, self.state_id())

    def set_verbosity(self, v): self.verbosity = v

    def __repr__(self):
        state = copy.deepcopy(self.state)
        state['hot'] = self.hot
        # state['contains'] = self.contains
        string = f"{self.name}: {str(state)}, contains: {str(self.contains)}"
        return string

    # ======================================================
    # Actions
    # ======================================================
    def can_pickup(self): return self.pickupable
    def accepts(self, object):
        return object.type in self.can_contain

    def slice(self, env, pos, carrying):
        can_slice = self.has_prop('sliced')
        # can't slice? action failed
        if not can_slice: return False
        # already sliced, failed
        if self.state['sliced']: return False

        if carrying.type == 'knife':
            self.set_prop("sliced", True)
            return True
        
        # wasn't carrying knife, failed
        return False


    def heat_contents(self):
        # not hot, don't heat contents
        if not self.hot: return False

        # no contents to apply to
        if self.contains is None: return False

        # apply recursively
        self.contains.hot = True
        self.contains.heat_contents()



    def clean_contents(self):
        # only apply to contents if on
        if not(self.has_prop('on') and self.state['on']):
            return

        if 'dirty' in self.state:
            self.state['dirty'] = False

        # no contents to apply to
        if self.contains is None: return

        # will NOT apply recursively
        if 'dirty' in self.contains.state:
            self.contains.state['dirty'] = False


    def toggle(self, env, pos):
        can_toggle = self.has_prop('on')
        # can't toggle, action fails
        if not can_toggle: return False

        # already toggled, action failed
        if self.state['on']:
            if self.can_heat and not self.hot:
                raise RuntimeError("This shouldn't happen")
            return False

            self.set_prop("on", True)

            # can heat? 
            if self.can_heat:
                self.hot = True

            # heat container objects
            if self.can_heat_contained:
                self.heat_contents()
            elif self.can_clean_contained:
                self.clean_contents()

            return True
        else:
            self.set_prop("on", False)
            self.hot = False
            return True




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

    def heat_contents(self):
        self.state['cooked'] = True


class KitchenContainer(KitchenObject):
    """docstring for KitchenContainer"""
    def __init__(self, hides_content=False, *args, **kwargs):
        super(KitchenContainer, self).__init__(*args, 
            is_container=True,
            **kwargs)
        self.hides_content = hides_content

        assert self.can_contain, "must accept things"


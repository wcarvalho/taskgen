from sfgen.babyai_kitchen.objects import KitchenObject, Food, KitchenContainer


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

        self.reset()

    @property
    def objects(self):
        return self._objects

    @staticmethod
    def _default_objects():
        return [
                KitchenContainer(
                    name="sink", 
                    properties=['on'],
                    can_contain=['knife', 'pot', 'pan', 'fork', 'plates'],
                    pickupable=False,
                    can_clean_contained=True,
                ),
                KitchenContainer(
                    name="stove", 
                    properties=['on'],
                    can_contain=['pot', 'pan'],
                    toggle_heats=True,
                    pickupable=False,
                    can_heat_contained=True,
                    can_heat=True,
                ),

                KitchenContainer(
                    name="pot", 
                    hides_content=True,
                    can_contain=['lettuce', 'potato', 'tomato', 'onion'],
                    properties=['dirty'],
                    can_heat_contained=True,
                    can_heat=True,
                ),
                KitchenContainer(
                    name="pan",
                    can_contain=['lettuce', 'potato', 'tomato', 'onion'],
                    hides_content=True,
                    properties=['dirty'],
                    can_heat_contained=True,
                    can_heat=True,
                ),
                KitchenContainer(
                    name="plates",
                    can_contain=['lettuce', 'potato', 'tomato', 'onion'],
                    properties=['dirty'],
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

    # ======================================================
    # environment functions
    # ======================================================
    def add_grid(self, grid):
        self.grid = grid

    def reset(self, randomize_states=False):
        self.last_action_information = {}
        for object in self.objects:
            object.reset_state(random=randomize_states)

    # ======================================================
    # Interacting in environment
    # ======================================================

    def place_inside(self, carrying, container):
        # not container, keep object
        if not container.is_container: return False, carrying

        # container doesn't accept the type being carried
        if not container.accepts(carrying): return False, carrying

        # container is full
        if container.contains is not None:
            # try recursively placing. e.g. if stove, to put in pot
            return self.place(carrying, container.contains)
            # return carrying


        # place object inside container
        container.contains = carrying
        carrying.cur_pos = np.array([-1, -1])

        # container has objects
        if container.can_heat_contained:
            # heat object
            container.heat_contents()
        elif container.can_clean_contained:
            container.clean_contents()

        # no longer have object
        carrying = None
        return True, carrying

if __name__ == '__main__':
    Food("knife")
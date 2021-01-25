from ipywidgets import interact, interactive
import ipywidgets as widgets
import glob
import os
from collections import OrderedDict
import types

def get_dirs(path):
    dirs = glob.glob(path)
    bases = [os.path.basename(e) for e in dirs]
    mapping = {b:f for b,f in zip(bases, dirs)}
    return dirs, bases, mapping

def create_fn(node1, node2):
    """Construct the dropdowns that will define the recursion to the end of the path"""

    fn  = """
def load_{0}(self, {0}):
    path = {0}

    # this maps the path to its full path
    if hasattr(self, '{0}_mapping'):
        path = self.{0}_mapping[path]
        # pprint(self.{0}_mapping)

    # this adds the "inner_path" to path
    # that lets you search inside the inner_path
    if '{2}' and not '{2}'.lower() == 'none':
        path = os.path.join(path, '{2}')

    # this creates both options for viewing from the dropdown
    # and the mapping which defines what full path corresponds to what option
    # i.e. its a dict like option:fullpath
    _, options, mapping = get_dirs(os.path.join(path, '*{4}*'))
    self.{1}_mapping = mapping
    options = sorted(options)
    
    # this adds a default value to the keyword arguments if it was given
    kwargs=dict()
    if ('{3}' and not '{3}'.lower() == 'none') and '{3}' in options:
       kwargs['value'] = '{3}'
    
    # create the dropdown
    self.{0}_dropdown = widgets.Dropdown(
        options=options,
        description="{0}:",
        **kwargs,
    )

    self.{0} = self.{0}_dropdown.value

    # display the dropdown (this will happen when this function is called)
    display(interactive(
        self.load_{1},
        {1}=self.{0}_dropdown, 
    ))
        """.format(node1["name"], node2.get("name", ""), node1.get("inner_path", ""), node1.get("default_value", ""), node1.get("regex", "*"))

    exec(fn)
    fn_name = "load_{0}".format(node1["name"])
    final_fn = locals()[fn_name]
    return final_fn, fn_name


class FileLoader(object):
    """docstring for ClassName"""
    def __init__(self, search_path, base_path, file_processor=None):
        self.file_processor = file_processor

        search_path.append(dict(name="final"))

        if len(search_path) < 1:
            raise NotImplementedError
        else:
            self.fns = OrderedDict()
            for i in range(len(search_path[:-1])):
                j = i + 1
                
                # create each recursive function
                fn, fn_name = create_fn(search_path[i], search_path[j])
                setattr(self, fn_name, types.MethodType(fn, self))
                self.fns[fn_name] = fn

            # create the final function which will get the final path
            setattr(self, fn_name, types.MethodType(fn, self))
            self.fns[fn_name] = fn
            first_fn = next(iter(self.fns))

            # call first function to start the process
            self.fns[first_fn](self, base_path)

    def load_final(self, final):
        if hasattr(self, 'final_mapping'):
            path = self.final_mapping[final]

        self.file = {0}
        self.fullpath = path

        print('------------Attributes Available:------------')
        print('Default')
        print('\tfile')
        print('\tfullpath')

        if self.file_processor:
            # this will be set by the final create_final_fn functino
            results = self.file_processor(self.fullpath)

            if not results: return
            # make all the results members of the class
            print('Loaded')
            for key, value in results.items():
                print('\t%s' % key)
                setattr(self, key, value)

import collections

# ======================================================
# Utilities for handling dictionaries
# ======================================================

def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def consolidate_dict_list(dict_list):
    consolidation = flatten_dict(dict_list[0], sep="/")
    consolidation = {k: [v] if not isinstance(v, list) else v for k,v in consolidation.items()}
    for next_dict in dict_list[1:]:
        fnext_dict = flatten_dict(next_dict, sep="/")
        for k, v in consolidation.items():
            newv = fnext_dict[k]
            if isinstance(newv, list):
                consolidation[k].extend(newv)
            else:
                consolidation[k].append(newv)

    return consolidation

def dictop(dictionary: dict, op, skip=[]):
  """Apply function recursively to dictionary
  
  Args:
      dictionary (dict): dict
      op (TYPE): function to apply
      skip (list, optional): keys to skip
  
  Returns:
      TYPE: Description
  """
  if not isinstance(dictionary, dict):
    try:
        return op(dictionary)
    except Exception as e:
        print(e)
        return None
  return {k: dictop(v, op) if (not (k is None or k in skip)) else v for k,v in dictionary.items()}

def multi_dictop(dictionaries : list, fn):
  default = dictionaries[0]
  keys = default.keys()
  output={}
  for key in keys:
    output[key] = fn([dictionary[key] for dictionary in dictionaries])

  return output

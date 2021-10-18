import collections
from utils.runners import record_tabular_misc_stat

def layoutname(fp):
  if fp >=1 and fp < 31: return "kitchen"
  elif fp >=200 and fp < 231: return "livingroom"
  elif fp >=300 and fp < 331: return "bedroom"
  elif fp >=400 and fp < 431: return "bathroom"
  else: raise RuntimeError

def distance_group(dist):
  if dist <=3: return "0.short"
  elif dist <=5: return "1.medium"
  else: return "2.far"

def thor_nav_log_fn(infos, desc: str=""):
  """Function for logging that stratifies data 
    by distance and floorplan.
  Args:
      infos (TYPE): list[ThorTrajInfo]
      desc (str): train, eval, etc.
  """
  # all keys to print out
  keys = list(filter(lambda k: not k.startswith("_"), infos[0].keys()))
  # function for tensorboard key
  tbkey = lambda a, k: f"{k}/{desc}_{a}" if desc else f"{k}/{a}"
  # ======================================================
  # print aggregated stats
  # ======================================================
  for k in keys:
    # all data for this key (e.g. "Length")
    data = [info[k] for info in infos]
    record_tabular_misc_stat(tbkey('0.all', k), data)

  # ======================================================
  # print stats grouped by (a) distances + (b) floorplans
  # ======================================================
  # get attribute names (e.g. "0.short_kitchen")
  attrs = []
  for info in infos:
    dist = distance_group(info['object_dist'])
    floorplan = layoutname(info['_floorplan'])
    key = f"{dist}_{floorplan}"
    attrs.append(key)

  # group data using dictionaries
  # {"0.short_kitchen": [info, info, ...]}
  attr2infos = collections.defaultdict(list)
  for attr, info in zip(attrs, infos):
    attr2infos[attr].append(info)

  # -----------------------
  # print grouped
  # -----------------------
  for attr, grouped_infos in attr2infos.items():
    record_tabular_misc_stat(tbkey(attr, 'ndata'), [len(grouped_infos)])
    for k in keys:
      # all data for this key (e.g. "Length")
      data = [info[k] for info in grouped_infos]
      # record together
      record_tabular_misc_stat(tbkey(attr, k), data)

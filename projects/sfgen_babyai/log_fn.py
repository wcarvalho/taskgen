import collections
from utils.runners import record_tabular_misc_stat

def kitchen_log_fn(infos, levelname2idx, desc: str=""):
  """Function for logging that stratifies bu levelname
  Args:
      infos (TYPE): list[BabyAITrajInfo]
      desc (str): train, eval, etc.
  """
  desc = desc or 'results'
  # all keys to print out
  keys = list(filter(lambda k: not k.startswith("_"), infos[0].keys()))

  idx2levelname = {idx:level for level, idx in levelname2idx.items()}


  # ======================================================
  # print aggregated stats
  # ======================================================
  for k in keys:
    # all data for this key (e.g. "Length")
    data = [info[k] for info in infos]
    record_tabular_misc_stat(f'{k}/{desc}_0.all', data)

  # ======================================================
  # print stats grouped by (a) levelname
  # ======================================================

  # group data using dictionaries
  level2infos = collections.defaultdict(list)
  for info in infos:
    level_idx = info['_task']
    level = idx2levelname[level_idx]
    level2infos[level].append(info)

  # -----------------------
  # print grouped
  # -----------------------
  for level, grouped_infos in level2infos.items():
    record_tabular_misc_stat(f'ndata/{desc}_{level}', [len(grouped_infos)])
    for k in keys:
      # all data for this key (e.g. "Length")
      data = [info[k] for info in grouped_infos]
      # record together
      record_tabular_misc_stat(
        f'{k}/{desc}_{level}',
        data)

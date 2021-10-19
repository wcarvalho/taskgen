import json
from tqdm import trange
from pprint import pprint
import babyai.utils

from envs.babyai_kitchen.levelgen import KitchenLevel
from envs.babyai_kitchen.tasks import TASKS

def main():
    # ======================================================
    # create object to store vocabulary
    # ======================================================
    instr_preproc = babyai.utils.format.InstructionsPreprocessor(path="preloads/babyai_kitchen/vocab.json")


    # ======================================================
    # load env
    # ======================================================
    env = KitchenLevel(
        task_kinds=list(TASKS.keys()),
        use_time_limit=False)

    for task, Cls in TASKS.items():
      instance = Cls(env.kitchen)
      mission = instance.abstract_rep
      instr_preproc([dict(mission=mission)])

    for object in env.kitchen.objects:
      instr_preproc([dict(mission=object.type)])


    pprint(instr_preproc.vocab.vocab)
    instr_preproc.vocab.save(verbosity=1)



if __name__ == "__main__":
    main()

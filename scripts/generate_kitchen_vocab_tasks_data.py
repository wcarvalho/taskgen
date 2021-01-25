import json
import sys, inspect
from tqdm import tqdm, trange

import babyai.utils

from sfgen.babyai_kitchen.levelgen import KitchenLevel

def main():
    # ======================================================
    # create object to store vocabulary
    # ======================================================
    instr_preproc = babyai.utils.format.InstructionsPreprocessor(model_name="babyai_kitchen")


    # ======================================================
    # load env
    # ======================================================
    env = KitchenLevel(
        task_kinds=[
            "clean",
            "slice",
            "cook",
            "cool",
            "heat",
            "place",
        ],
        load_actions_from_tasks=True,
        use_time_limit=False)

    task2idx = {}
    # ======================================================
    # loop through each level, gen random seed and try to add
    # ======================================================
    t1 = trange(int(1e4), desc='', leave=True)
    for step in t1:
        obs = env.reset()
        instr_preproc([obs])
        t1.set_description(obs['mission'])
        task2idx[obs['mission']] = len(task2idx)

    instr_preproc.vocab.save(verbosity=1)
    with open('models/babyai_kitchen/tasks.json', "w") as f:
        json.dump(task2idx, f)



if __name__ == "__main__":
    main()

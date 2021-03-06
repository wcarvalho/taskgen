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
    t1 = trange(int(5e4), desc='', leave=True)
    for step in t1:
        obs = env.reset()
        instr = obs['mission']
        t1.set_description(instr)


        instr_preproc([obs])
        if instr not in task2idx:
            task2idx[instr] = len(task2idx) + 1

    # instr_preproc.vocab.save(verbosity=1)
    file='./models/babyai_kitchen/tasks.json'
    with open(file, "w") as f:
        json.dump(task2idx, f)
        print(f"Saved {file}")



if __name__ == "__main__":
    main()

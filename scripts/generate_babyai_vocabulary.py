import json
import sys, inspect
import babyai.utils
from tqdm import tqdm, trange

def main():
    # ======================================================
    # create object to store vocabulary
    # ======================================================
    instr_preproc = babyai.utils.format.InstructionsPreprocessor(path="preloads/babyai/vocab.json")

    # ======================================================
    # get all levels
    # ======================================================
    env_classes = []
    for name, obj in inspect.getmembers(babyai.levels.iclr19_levels):
        if inspect.isclass(obj) and "Level" in name:
            env_classes.append(obj)

    task2idx = {}
    # ======================================================
    # loop through each level, gen random seed and try to add
    # ======================================================
    t1 = tqdm(env_classes, desc='classes')
    for env_class in t1:
        try:
            env = env_class()
        except NotImplementedError as ne:
            continue
        except Exception as e:
            raise e
        t1.set_description(str(env_class))
        t2 = trange(1000, desc='', leave=True)
        for step in t2:
            obs = env.reset()
            instr_preproc([obs])
            instr = obs['mission']
            if instr not in task2idx:
                task2idx[instr] = len(task2idx)
            t2.set_description(instr)

    instr_preproc.vocab.save(verbosity=1)
    file='./preloads/babyai/tasks.json'
    with open(file, "w") as f:
        json.dump(task2idx, f)
        print(f"Saved {file}")


if __name__ == "__main__":
    main()

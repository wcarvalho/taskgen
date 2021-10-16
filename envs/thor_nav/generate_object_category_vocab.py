import numpy as np
import json
import sys, inspect
from tqdm import tqdm, trange

import ai2thor.controller

def main():
  vocab = {}
  all_floorplans = [*np.arange(1,31), *np.arange(201, 231), *np.arange(301, 331), *np.arange(401, 431)]


  conv_kwargs=dict(
    action='Initialize',
    visibilityDistance=1.5,
    scene="FloorPlan212",

    # step sizes
    gridSize=0.25,
    snapToGrid=True,
    rotateStepDegrees=90,

    # image modalities
    renderDepthImage=False,
    renderInstanceSegmentation=False,

    # camera properties
    width=300,
    height=300,
    fieldOfView=90
    )
  controller = ai2thor.controller.Controller()
  controller.start()


  for fp in all_floorplans:
    import ipdb; ipdb.set_trace()
    event = controller.reset(f"FloorPlan{fp}")
    event = controller.step(**conv_kwargs)

  controller.stop()
    # for obj_name, info in env.objects_by_name_dict.items():
    #   obj_type = info['objectType']
    #   if obj_type not in all_object_infos:
    #     all_object_infos[obj_type] = info
    #     print(f'New object: {obj_type}')

    # env.close()

  # output_file = args['object_info_file']
  # basedir = os.path.dirname(output_file)
  # path_exists(basedir)
  # pickle.dump(all_object_infos, open(output_file, 'wb'))

  # ======================================================
  # create object to store vocabulary
  # ======================================================
  # instr_preproc = babyai.utils.format.InstructionsPreprocessor(model_name="babyai")

  # # ======================================================
  # # get all levels
  # # ======================================================
  # env_classes = []
  # for name, obj in inspect.getmembers(babyai.levels.iclr19_levels):
  #     if inspect.isclass(obj) and "Level" in name:
  #         env_classes.append(obj)

  # task2idx = {}
  # # ======================================================
  # # loop through each level, gen random seed and try to add
  # # ======================================================
  # t1 = tqdm(env_classes, desc='classes')
  # for env_class in t1:
  #     try:
  #         env = env_class()
  #     except NotImplementedError as ne:
  #         continue
  #     except Exception as e:
  #         raise e
  #     t1.set_description(str(env_class))
  #     t2 = trange(1000, desc='', leave=True)
  #     for step in t2:
  #         obs = env.reset()
  #         instr_preproc([obs])
  #         instr = obs['mission']
  #         if instr not in task2idx:
  #             task2idx[instr] = len(task2idx)
  #         t2.set_description(instr)

  # instr_preproc.vocab.save(verbosity=1)
  # file='./preloads/babyai/tasks.json'
  # with open(file, "w") as f:
  #     json.dump(task2idx, f)
  #     print(f"Saved {file}")


if __name__ == "__main__":
    main()

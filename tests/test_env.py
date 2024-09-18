import os, sys
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import hydra
import copy, json
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from termcolor import colored
from omegaconf import OmegaConf
from moviepy.editor import ImageSequenceClip
from collections import Counter, defaultdict, namedtuple

sys.path.append('/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/yanfeng/project/robotic/calvin/calvin_models')
sys.path.append('/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/yanfeng/project/robotic/calvin/calvin_env')
sys.path.append('/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/yanfeng/project/robotic/calvin/calvin_env/tacto_env')
sys.path.append('/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/yanfeng/project/robotic/RoboFlamingo/open_flamingo')
from calvin_env.envs.play_table_env import get_env
from calvin_agent.evaluation.utils import (
    count_success,
    get_env_state_for_initial_condition,
    print_and_save,
)

# import pyrender
EP_LEN = 360
NUM_SEQUENCES = 1000

def make_env(dataset_path):
    val_folder = Path(dataset_path) / "validation"
    env = get_env(val_folder, show_gui=False)

    # insert your own env wrapper
    # env = Wrapper(env)
    return env

def evaluate_sequence(env, model, task_checker, initial_state, eval_sequence, val_annotations, plans, debug, eval_log_dir='', sequence_i=-1, reset=False, diverse_inst=False):
    """
    Evaluates a sequence of language instructions.
    """
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

    success_counter = 0
    for subtask_i, subtask in enumerate(eval_sequence):
        if reset:
            success = rollout(env, model, task_checker, subtask, val_annotations, plans, debug, eval_log_dir, subtask_i, sequence_i, robot_obs=robot_obs, scene_obs=scene_obs, diverse_inst=diverse_inst)
        else:
            success = rollout(env, model, task_checker, subtask, val_annotations, plans, debug, eval_log_dir, subtask_i, sequence_i,diverse_inst=diverse_inst)
        if success:
            success_counter += 1
        else:
            return success_counter
    return success_counter

def rollout(env, model, task_oracle, subtask, val_annotations, plans, debug, eval_log_dir='', subtask_i=-1, sequence_i=-1, robot_obs=None, scene_obs=None, diverse_inst=False):
    """
    Run the actual rollout on one subtask (which is one natural language instruction).
    """
    planned_actions = []
    if robot_obs is not None and scene_obs is not None:
        env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
    obs = env.get_obs()
    # get lang annotation for subtask
    if diverse_inst:
        lang_annotation = val_annotations[sequence_i][subtask_i]
    else:
        lang_annotation = val_annotations[subtask][0]
    lang_annotation = lang_annotation.split('\n')[0]
    if '\u2019' in lang_annotation:
        lang_annotation.replace('\u2019', '\'')
    # model.reset()
    start_info = env.get_info()

    img_queue = []
    for step in range(EP_LEN):
        
        # action = model.step(obs, lang_annotation, (len(planned_actions) == 0))
        action = np.array([0.1, 0.1, 0.1, 0, 0, 0, 1])
        if len(planned_actions) == 0:
            if action.shape == (7,):
                planned_actions.append(action)
            else:
                planned_actions.extend([action[i] for i in range(action.shape[0])])
        action = planned_actions.pop(0)
        obs, _, _, current_info = env.step(action)
        img_copy = copy.deepcopy(obs['rgb_obs']['rgb_static'])
        img_queue.append(img_copy)
        # if step == 0:
        #     # for tsne plot, only if available
        #     collect_plan(model, plans, subtask)

        # check if current step solves a task
        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
        if len(current_task_info) > 0:
            print(colored("success", "green"), end=" ")
            img_clip = ImageSequenceClip(img_queue, fps=30)
            img_clip.write_gif(os.path.join(eval_log_dir, f'{sequence_i}-{subtask_i}-{subtask}-succ.gif'), fps=30)
            return True

    print(colored("fail", "red"), end=" ")
    img_clip = ImageSequenceClip(img_queue, fps=30)
    img_clip.write_gif(os.path.join(eval_log_dir, f'{sequence_i}-{subtask_i}-{subtask}-fail.gif'), fps=30)
    return False


dataset_path = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/yanfeng/data/robotics/calvin/"
calvin_conf_path = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/yanfeng/project/robotic/calvin/calvin_models/conf"
eval_log_dir = 'exps/tmp'
debug=True
reset=False
diverse_inst=False
model = None
env = make_env(dataset_path)


subtask = 'lift_red_block_table'

conf_dir = Path(calvin_conf_path) # 导入环境基本信息
task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
task_oracle = hydra.utils.instantiate(task_cfg)

if diverse_inst:
    with open('/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/yanfeng/project/robotic/RoboFlamingo2/lang_annotation_cache.json', 'r') as f:
        val_annotations = json.load(f)
else:
    val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")


eval_log_dir = eval_log_dir
os.makedirs(eval_log_dir, exist_ok=True)
with open('/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/yanfeng/project/robotic/RoboFlamingo2/eval_sequences.json', 'r') as f:
    eval_sequences = json.load(f)

results = []
plans = defaultdict(list)
local_sequence_i = 0
eval_sequences = tqdm(eval_sequences, position=0, leave=True)
for initial_state, eval_sequence in eval_sequences:
    result = evaluate_sequence(env, model, task_oracle, initial_state, eval_sequence, val_annotations, plans, debug, eval_log_dir, local_sequence_i, reset=reset, diverse_inst=diverse_inst)
    results.append(result)
    if not debug:
        eval_sequences.set_description(
            " ".join([f"{i + 1}/5 : {v * 100:.1f}% |" for i, v in enumerate(count_success(results))]) + "|"
        )
    local_sequence_i += 1

def extract_iter_from_tqdm(tqdm_iter):
        return [_ for _ in tqdm_iter]

eval_sequences = extract_iter_from_tqdm(eval_sequences)

res_tup_list = [(res, eval_seq) for res, eval_seq in zip(results, eval_sequences)]

res_list = [_[0] for _ in res_tup_list]
eval_seq_list = [_[1] for _ in res_tup_list]
print_and_save(res_list, eval_seq_list, eval_log_dir, 0)


# val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml") # 导入语言标注
# lang_annotation = val_annotations[subtask][0]
# lang_annotation = lang_annotation.split('\n')[0]
# if '\u2019' in lang_annotation:
#     lang_annotation.replace('\u2019', '\'')

# initial_state = {'led': 0, 'lightbulb': 0, 'slider': 'right', 'drawer': 'closed', 'red_block': 'table', 'blue_block': 'slider_right', 'pink_block': 'slider_left', 'grasped': 0}
# robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)

# success_counter = 0
# task_num = 20
# EP_LEN = 360 
# bound_restriction = {
#     "table_bound_x_min":-0.05,
#     "table_bound_x_max":0.3,
#     "table_bound_y_max":-0.035,
#     "table_bound_y_min":-0.135,
# }

# for task_i in range(task_num):
#     random_position = np.array([np.random.uniform(bound_restriction['table_bound_x_min'], bound_restriction['table_bound_x_max']),
#                                 np.random.uniform(bound_restriction['table_bound_y_min'], bound_restriction['table_bound_y_max']),
#                                 4.59990010e-01])
#     scene_obs[6:9] = random_position
#     success = False # determine success or not
#     env.reset(robot_obs=robot_obs, scene_obs=scene_obs)  # reset the environment eachtime
#     obs = env.get_obs()
#     start_info = env.get_info()

#     planned_actions = []
#     img_queue = [] # save frames for the .gif file

#     for step in range(EP_LEN):
        
#         action = np.array([0.1, 0.1, 0.1, 0, 0, 0, 1])

#         obs, _, _, current_info = env.step(action)

#         img_copy = copy.deepcopy(obs['rgb_obs']['rgb_static']) # s
#         img_queue.append(img_copy)

#         # check if current step solves a task
#         current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
#         if len(current_task_info) > 0:
#             print(colored("success", "green"), end=" ")
#             img_clip = ImageSequenceClip(img_queue, fps=30)
#             img_clip.write_gif(os.path.join(eval_log_dir, f'{subtask}-{task_i}-succ.gif'), fps=30)
#             success = True
#             break
#     if not success:
#         print(colored("fail", "red"), end=" ")
#         img_clip = ImageSequenceClip(img_queue, fps=30)
#         img_clip.write_gif(os.path.join(eval_log_dir, f'{subtask}-{task_i}-fail.gif'), fps=30)
#     else:
#         success_counter += 1

# results = {
#     'task_instruction': lang_annotation,
#     'success_counter': success_counter,
#     'task_num':task_num,
#     'success_rate':success_counter/task_num
# }

# with open(os.path.join(eval_log_dir,'results.json'), 'w') as json_file:
#     json.dump(results, json_file)

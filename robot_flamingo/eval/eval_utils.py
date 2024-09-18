import argparse
from collections import Counter, defaultdict, namedtuple
import logging
import os, json, random
from pathlib import Path
import sys
import time
import PIL.Image as Image
import copy
from collections import deque
import torch.distributed as dist
from moviepy.editor import ImageSequenceClip
# This is for using the locally installed repo clone when using slurm
from calvin_agent.models.calvin_base_model import CalvinBaseModel
from typing import Optional
import time
import numpy as np

from calvin_agent.evaluation.multistep_sequences import get_sequences
from calvin_agent.evaluation.utils import (
    collect_plan,
    count_success,
    create_tsne,
    get_default_model_and_env,
    get_env_state_for_initial_condition,
    get_log_dir,
    join_vis_lang,
    print_and_save,
)
from calvin_agent.utils.utils import get_all_checkpoints, get_checkpoints_for_epochs, get_last_checkpoint
import hydra
import numpy as np
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from termcolor import colored
import torch
from torch.nn.parallel import DistributedDataParallel
from tqdm.auto import tqdm
from calvin_env.envs.play_table_env import get_env
from robot_flamingo.data.data import preprocess_image, preprocess_text_calvin
from robot_flamingo.utils import world_to_tcp_frame, tcp_to_world_frame
import functools

# os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
# import pyrender
logger = logging.getLogger(__name__)

EP_LEN = 360
NUM_SEQUENCES = 1000


# NUM_SEQUENCES = 400

def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == "bf16" or precision == "amp_bf16":
        cast_dtype = torch.bfloat16
    elif precision == "fp16":
        cast_dtype = torch.float16
    return cast_dtype


def make_env(dataset_path):
    val_folder = Path(dataset_path) / "validation"
    env = get_env(val_folder, show_gui=False)

    # insert your own env wrapper
    # env = Wrapper(env)
    return env


class DebugEnv():

    def __init__(self) -> None:
        pass

    def get_random_obs(self):
        obs = {}
        obs['rgb_obs'] = {}
        obs['rgb_obs']['rgb_static'] = np.ones((200, 200, 3), dtype=np.uint8)
        obs['rgb_obs']['rgb_gripper'] = np.ones((84, 84, 3), dtype=np.uint8)
        obs['robot_obs'] = np.ones(15, dtype=np.float32)
        return obs

    def get_obs(self):
        return self.get_random_obs()

    def step(self, action):
        return self.get_random_obs()

    def reset(self, **kwargs):
        return

    def get_info(self):
        return


def make_env_debug(dataset_path):
    val_folder = Path(dataset_path) / "validation"
    env = get_env(val_folder, show_gui=False)

    # insert your own env wrapper
    # env = Wrapper(env)
    return env


class ModelWrapper(CalvinBaseModel):
    def __init__(self, model, tokenizer, image_processor, cast_dtype, use_diff, history_len=None, future_act_len=-1):
        super().__init__()
        self.model = model
        self.replan = model.module.replan
        self.decoder_type = model.module.decoder_type
        self.cast_type = cast_dtype
        self.use_diff = use_diff
        self.text_process_fn = functools.partial(preprocess_text_calvin, tokenizer=tokenizer)
        self.image_process_fn = functools.partial(preprocess_image, image_processor=image_processor)
        self.action_hist_queue = []
        self.feature_cache = None
        self.dt_feat_cache = []
        self.fusion_mode = self.model.module.fusion_mode

        self.avg_step = 0
        self.infer_time = 1.0

        # # 先指定某一类任务
        # self.bounds_dict = get_gripper_loc_bounds(task="push_buttons")

        if use_diff:
            self.diffusion_model = None
            self.normalizer = None
            if isinstance(self.model, DistributedDataParallel):
                self.diffusion_model = self.model.module.diffusion_model
            else:
                self.diffusion_model = self.model.diffusion_model
            action_dim = self.diffusion_model.data_dim
            horizon = self.diffusion_model.horizon
            self.normalizer = self.diffusion_model.normalizer
            self.action_hist_queue = deque(maxlen=history_len - 1)
            self.action_hist_queue.extend([np.zeros(action_dim) for _ in range(history_len - 1)])

            if horizon - history_len + 1:
                self.supp = None
            self.hist_len = history_len - 1
            self.action_dim = action_dim
            self.horizon = horizon
            self.future_act_len = future_act_len

        # if self.model.module.pad_length != -1:
        if self.model.module.pad_length == -1:
            history_len = self.model.module.window_size
        self.img_queue = deque(maxlen=history_len)
        self.gripper_queue = deque(maxlen=history_len)
        self.state_queue = deque(maxlen=history_len)
        self.mask_queue = deque(maxlen=history_len)
        self.text_queue = deque(maxlen=history_len)

    def reset(self):
        """
        This is called
        """
        if self.use_diff:
            self.action_hist_queue = deque(maxlen=self.hist_len)
            self.action_hist_queue.extend([np.zeros(self.action_dim) for _ in range(self.hist_len)])
        if self.model.module.pad_length != -1:
            history_len = self.model.module.pad_length
        else:
            history_len = self.model.module.window_size
        self.img_queue = deque(maxlen=history_len)
        self.gripper_queue = deque(maxlen=history_len)
        self.state_queue = deque(maxlen=history_len)
        self.mask_queue = deque(maxlen=history_len)
        self.text_queue = deque(maxlen=history_len)
        self.feature_cache = None
        self.dt_feat_cache = []

        self.model.module.lang_encoder.lm_head.hidden_state = None
        self.model.module.lang_encoder.lm_head.history_memory = []

        if self.model.module.sep_lm_head:
            self.model.module.lm_head.hidden_state = None
            self.model.module.lm_head.history_memory = []

    def step(self, obs, goal, get_action=True):
        """
        Args:
            obs: environment observations
            goal: embedded language goal
        Returns:
            action: predicted action
        """
        # 如果不需要获取action直接返回None
        hidden_inference = False  # 不使用mask输入
        if not get_action:
            if self.model.module.mask_ratio > 0:
                hidden_inference = True  # 使用mask输入
            else:
                return None
        if not hidden_inference or self.model.module.use_vit:
            image = obs["rgb_obs"]['rgb_static']
            image = Image.fromarray(image)
            image_x = self.image_process_fn([image])
            # expand image dimension
            image_x = image_x.unsqueeze(1).unsqueeze(1).to(dtype=self.cast_type)
            # expand text dimension
            text_x, mask = self.text_process_fn([goal])
        else:
            image_x = torch.randn(1)
            text_x = torch.randn(1)
            mask = torch.randn(1)

        # fix window_size : ddp_model -> ... -> window_size
        if self.model.module.sep_lm_head:
            window_size = self.model.module.lm_head.window_size
            self.model.module.lm_head.window_size = 1
            if self.model.module.pad_length != -1 and self.feature_cache is None:
                self.model.module.lm_head.window_size = self.model.module.pad_length
        else:
            window_size = self.model.module.lang_encoder.lm_head.window_size
            self.model.module.lang_encoder.lm_head.window_size = 1
            if self.model.module.pad_length != -1 and self.feature_cache is None:
                self.model.module.lang_encoder.lm_head.window_size = self.model.module.pad_length
        gripper = None
        state = None

        if self.model.module.use_gripper and not hidden_inference:
            gripper = obs["rgb_obs"]['rgb_gripper']
            gripper = Image.fromarray(gripper)
            gripper = self.image_process_fn([gripper])
            # expand image dimension
            gripper = gripper.unsqueeze(1).unsqueeze(1).to(dtype=self.cast_type)
        else:
            gripper = torch.randn(1)

        # if self.model.module.use_state or self.model.module.sep_lm_head:
        if self.model.module.use_state or self.model.module.sep_lm_head:
            state = obs['robot_obs']
            state = torch.from_numpy(np.stack([state]))
            # if self.model.module.sep_lm_head:
            #     state = torch.cat([state[...,:6], state[...,[-1]]], dim=-1)
            if self.fusion_mode == 'two_way':
                state = state.repeat(2, 1)
            state = state.unsqueeze(1).unsqueeze(1).to(dtype=self.cast_type)
            state = state.to(torch.float32)
        with torch.no_grad():
            device = 'cuda'
            image_x = image_x.to(device)
            text_x = text_x.to(device)
            mask = mask.to(device)
            if gripper is not None:
                gripper = gripper.to(device)
            if state is not None:
                state = state.to(device)

            # if self.model.module.pad_length != -1:
            if len(self.img_queue) == 0:
                self.img_queue.append(image_x)
                for _ in range(self.model.module.pad_length - 1):
                    self.img_queue.append(image_x)
            else:
                self.img_queue.append(image_x)
            if len(self.gripper_queue) == 0 and gripper is not None:
                self.gripper_queue.append(gripper)
                for _ in range(self.model.module.pad_length - 1):
                    self.gripper_queue.append(gripper)
            else:
                self.gripper_queue.append(gripper)
            if len(self.state_queue) == 0 and state is not None:
                self.state_queue.append(state)
                for _ in range(self.model.module.pad_length - 1):
                    self.state_queue.append(state)
            else:
                self.state_queue.append(state)
            if len(self.mask_queue) == 0 and mask is not None:
                self.mask_queue.append(mask)
                for _ in range(self.model.module.pad_length - 1):
                    self.mask_queue.append(mask)
            if len(self.text_queue) == 0 and text_x is not None:
                self.text_queue.append(text_x)
                for _ in range(self.model.module.pad_length - 1):
                    self.text_queue.append(text_x)

            if self.model.module.pad_length != -1 and self.feature_cache is None:
                image_x = torch.cat(list(self.img_queue), dim=0)
                if gripper is not None:
                    gripper = torch.cat(list(self.gripper_queue), dim=0)
                if state is not None:
                    state = torch.cat(list(self.state_queue), dim=0)
                mask = torch.cat(list(self.mask_queue), dim=0)
                text_x = torch.cat(list(self.text_queue), dim=0)

            if self.fusion_mode == 'vit_concat':
                image_x = torch.cat(list(self.img_queue), dim=0)
                if gripper is not None:
                    gripper = torch.cat(list(self.gripper_queue), dim=0)
                if state is not None:
                    state = torch.cat(list(self.state_queue), dim=0)
                pass

            if self.use_diff:
                if self.fusion_mode == 'two_way':
                    vision_x = torch.cat([image_x, gripper], dim=0)
                    text_x = text_x.repeat(2, 1)
                    mask = mask.repeat(2, 1)
                    model_out = self.model(vision_x=vision_x, lang_x=text_x, attention_mask=mask, state_tensor=state,
                                           return_feature=True)
                else:
                    model_out = self.model(vision_x=image_x, lang_x=text_x, attention_mask=mask, vision_gripper=gripper,
                                           state_tensor=state, return_feature=True)

                model_out = model_out.logits
                action_history = torch.tensor(np.stack(self.action_hist_queue, axis=0), dtype=torch.float,
                                              device=device).unsqueeze(0)
                action_history = self.normalizer.normalize(action_history)
                if self.supp is None:
                    self.supp = torch.zeros(
                        action_history.shape[0], self.horizon - self.hist_len, action_history.shape[-1],
                        dtype=action_history.dtype,
                        device=action_history.device,
                    )
                action_history = torch.concat([action_history, self.supp], dim=1)
                act_mask = torch.zeros_like(action_history, device=action_history.device, dtype=torch.bool)
                act_mask[:, :self.hist_len, ...] = 1.
                pred_action_seq = self.diffusion_model.conditional_sample(cond_data=action_history, cond_mask=act_mask,
                                                                          global_cond=model_out)
                pred_action_seq = self.normalizer.unnormalize(pred_action_seq)
                action = pred_action_seq[:, self.hist_len:, :]
                if self.future_act_len > 0:
                    action = action[:, :self.future_act_len, :]
                action = action[0]

                action = action.cpu().detach().numpy()
                action[..., -1] = action[..., -1] > 0.5
                action[..., -1] = (action[..., -1] - 0.5) * 2  # scale to -1 or 1
            else:
                if self.fusion_mode == 'two_way':
                    vision_x = torch.cat([image_x, gripper], dim=0)
                    text_x = text_x.repeat(2, 1)
                    mask = mask.repeat(2, 1)
                    action = self.model(vision_x=vision_x, lang_x=text_x, attention_mask=mask, state_tensor=state,
                                        return_feature=True, hidden_inference=hidden_inference)
                else:
                    action = self.model(vision_x=image_x, lang_x=text_x, attention_mask=mask, vision_gripper=gripper,
                                        state_tensor=state, return_feature=True, hidden_inference=hidden_inference)
                    if hidden_inference:
                        return None
                if self.model.module.pad_length != -1:
                    if self.feature_cache is None:
                        self.feature_cache = action.logits[-1]
                    else:
                        new_feat = torch.cat([self.feature_cache[1:], action.logits[-1]], dim=0)
                        self.feature_cache = new_feat
                        if not self.model.module.sep_lm_head:
                            self.model.module.lang_encoder.lm_head.window_size = window_size
                            lm_out = self.model.module.lang_encoder.lm_head(new_feat)
                        else:
                            self.model.module.lm_head.window_size = window_size
                            lm_out = self.model.module.lm_head(new_feat)
                        Output = namedtuple('Output', ['logits'])
                        action = Output(lm_out)

                if self.model.module.act_step == 1:
                    action = torch.concat((action.logits[0], action.logits[1] > 0.5), dim=2).squeeze(0)[
                        -1]  # support multi step history]
                    if self.model.module.use_waypoint or self.model.module.adaptive:
                        if action[6] > 0.5:
                            action[6] = torch.tensor(1).cuda()
                        else:
                            action[6] = torch.tensor(-1).cuda()

                else:
                    if "param" in self.model.module.episode_loss:
                        par_dict = action.logits[0]
                        parname_list = ['x_par', 'y_par', 'z_par', 'x_eul', 'y_eul', 'z_eul']
                        gripper = action.logits[1] > 0.5
                        bs, seq_len = par_dict['x_par'].shape[:2]
                        for parname in parname_list:
                            par_dict[parname] = par_dict[parname].view(bs * seq_len, -1)
                        time_var = torch.arange(self.model.module.act_step).cuda().float()
                        time_matrix = torch.cat([time_var ** 0, time_var, time_var ** 2, time_var ** 3], dim=0).reshape(
                            4,
                            self.model.module.act_step)
                        for parname in parname_list:
                            par_dict[parname] = torch.mm(par_dict[parname], time_matrix).view(bs, seq_len, -1)

                        pose = torch.stack([par_dict[par] for par in parname_list], dim=-1)

                    else:
                        pose = action.logits[0]
                        gripper = action.logits[1] > 0.5
                    if "sum" in self.model.module.episode_loss:
                        pose[..., 1:, :] = pose[..., 1:, :] - pose[..., :-1, :]  # 先前的action预测做了sum这里减掉
                    pose = pose.squeeze(0)[-1].view(self.model.module.act_step, -1)
                    gripper = gripper.squeeze(0)[-1].view(self.model.module.act_step, -1)
                    gripper = torch.where(gripper, torch.tensor(1).cuda(), torch.tensor(-1).cuda()).cuda()
                    action = torch.cat([pose, gripper], dim=-1)
                    if self.model.module.random_number:
                        self.model.module.action_num = random.randint(1, 5)
                    if self.model.module.use_waypoint or self.model.module.adaptive:
                        if self.model.module.adaptive == "feng":
                            self.model.module.action_num = get_waypoint_index(action,self.model.module.threshold)

                        self.avg_step = self.avg_step * (self.infer_time - 1) + self.model.module.action_num
                        self.avg_step = self.avg_step / self.infer_time
                        self.infer_time = self.infer_time + 1
                    action = action[:self.model.module.action_num]  # select first step action
                    action = action.cpu().detach().to(dtype=torch.float16).numpy()

        if self.model.module.sep_lm_head:
            self.model.module.lm_head.window_size = window_size
        else:
            self.model.module.lang_encoder.lm_head.window_size = window_size
        if self.model.module.tcp_rel:
            state = obs['robot_obs']
            state = torch.from_numpy(np.stack([state])).unsqueeze(0).float().cpu().detach()
            action = torch.from_numpy(np.stack([action])).unsqueeze(0).float().cpu().detach()
            action = tcp_to_world_frame(action, state)
            action = action.squeeze().to(dtype=torch.float16).numpy()
        return action


def evaluate_policy(model, env, epoch, calvin_conf_path, eval_log_dir=None, debug=False, create_plan_tsne=False,
                    diverse_inst=False):
    """
    Run this function to evaluate a model on the CALVIN challenge.

    Args:
        model: Must implement methods of CalvinBaseModel.
        env: (Wrapped) calvin env.
        epoch:
        eval_log_dir: Path where to log evaluation results. If None, logs to /tmp/evaluation/
        debug: If True, show camera view and debug info.
        create_plan_tsne: Collect data for TSNE plots of latent plans (does not work for your custom model)

    Returns:
        Dictionary with results
    """
    conf_dir = Path(calvin_conf_path)
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    if diverse_inst:
        with open(
                '/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/yanfeng/project/robotic/RoboFlamingo/enrich_lang_annotations.json',
                'r') as f:
            val_annotations = json.load(f)
    else:
        val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")

    eval_log_dir = get_log_dir(eval_log_dir)

    eval_sequences = get_sequences(NUM_SEQUENCES)

    results = []
    plans = defaultdict(list)

    if not debug:
        eval_sequences = tqdm(eval_sequences, position=0, leave=True)

    for initial_state, eval_sequence in eval_sequences:
        result = evaluate_sequence(env, model, task_oracle, initial_state, eval_sequence, val_annotations, plans, debug)
        results.append(result)
        if not debug:
            eval_sequences.set_description(
                " ".join([f"{i + 1}/5 : {v * 100:.1f}% |" for i, v in enumerate(count_success(results))]) + "|"
            )

    if create_plan_tsne:
        create_tsne(plans, eval_log_dir, epoch)

    print_and_save(results, eval_sequences, eval_log_dir, epoch)

    return results


def evaluate_policy_ddp(model, env, epoch, calvin_conf_path, eval_log_dir=None, debug=False, create_plan_tsne=False,
                        reset=False, diverse_inst=False, model_awe=None):
    """
    Run this function to evaluate a model on the CALVIN challenge.

    Args:
        model: Must implement methods of CalvinBaseModel.
        env: (Wrapped) calvin env.
        epoch:
        eval_log_dir: Path where to log evaluation results. If None, logs to /tmp/evaluation/
        debug: If True, show camera view and debug info.
        create_plan_tsne: Collect data for TSNE plots of latent plans (does not work for your custom model)

    Returns:
        Dictionary with results
    """
    conf_dir = Path(calvin_conf_path)
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)

    # val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")
    if diverse_inst:
        with open(
                '/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/yanfeng/project/robotic/RoboFlamingo2/lang_annotation_cache.json',
                'r') as f:
            val_annotations = json.load(f)
    else:
        val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")

    eval_log_dir = get_log_dir(eval_log_dir)
    with open(
            '/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/yanfeng/project/robotic/RoboFlamingo2/eval_sequences.json',
            'r') as f:
        eval_sequences = json.load(f)
    device_num = int(torch.distributed.get_world_size())
    device_id = torch.distributed.get_rank()
    assert NUM_SEQUENCES % device_num == 0
    interval_len = int(NUM_SEQUENCES // device_num)
    eval_sequences = eval_sequences[device_id * interval_len:min((device_id + 1) * interval_len, NUM_SEQUENCES)]
    results = []
    plans = defaultdict(list)
    local_sequence_i = 0
    base_sequence_i = device_id * interval_len

    if not debug:
        eval_sequences = tqdm(eval_sequences, position=0, leave=True)

    for initial_state, eval_sequence in eval_sequences:
        result = evaluate_sequence(env, model, task_oracle, initial_state, eval_sequence, val_annotations, plans, debug,
                                   eval_log_dir, base_sequence_i + local_sequence_i, reset=reset,
                                   diverse_inst=diverse_inst, model_awe=model_awe)
        results.append(result)
        if not debug:
            eval_sequences.set_description(
                f"{device_id}/{device_num}|" + " ".join(
                    [f"{i + 1}/5 : {v * 100:.1f}% |" for i, v in enumerate(count_success(results))]) + "|"
            )
        local_sequence_i += 1

    def merge_multi_list(res):
        tmp = []
        for l in res:
            tmp.extend(l)
        return tmp

    def extract_iter_from_tqdm(tqdm_iter):
        return [_ for _ in tqdm_iter]

    if create_plan_tsne:
        create_tsne(plans, eval_log_dir, epoch)

    eval_sequences = extract_iter_from_tqdm(eval_sequences)

    res_tup = [(res, eval_seq) for res, eval_seq in zip(results, eval_sequences)]
    all_res_tup = [copy.deepcopy(res_tup) for _ in range(device_num)] if torch.distributed.get_rank() == 0 else None
    torch.distributed.gather_object(res_tup, all_res_tup, dst=0)

    if torch.distributed.get_rank() == 0:
        res_tup_list = merge_multi_list(all_res_tup)
        res_list = [_[0] for _ in res_tup_list]
        eval_seq_list = [_[1] for _ in res_tup_list]
        print_and_save(res_list, eval_seq_list, eval_log_dir, epoch)

    return results


def evaluate_sequence(env, model, task_checker, initial_state, eval_sequence, val_annotations, plans, debug,
                      eval_log_dir='', sequence_i=-1, reset=False, diverse_inst=False, model_awe=None):
    """
    Evaluates a sequence of language instructions.
    """
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

    success_counter = 0
    if debug:
        time.sleep(1)
        print()
        print()
        print(f"Evaluating sequence: {' -> '.join(eval_sequence)}")
        print("Subtask: ", end="")
    for subtask_i, subtask in enumerate(eval_sequence):
        if reset:
            success = rollout(env, model, task_checker, subtask, val_annotations, plans, debug, eval_log_dir, subtask_i,
                              sequence_i, robot_obs=robot_obs, scene_obs=scene_obs, diverse_inst=diverse_inst,
                              model_awe=model_awe)
        else:
            success = rollout(env, model, task_checker, subtask, val_annotations, plans, debug, eval_log_dir, subtask_i,
                              sequence_i, diverse_inst=diverse_inst, model_awe=model_awe)
        if success:
            success_counter += 1
        else:
            return success_counter
    return success_counter


def rollout(env, model, task_oracle, subtask, val_annotations, plans, debug, eval_log_dir='', subtask_i=-1,
            sequence_i=-1, robot_obs=None, scene_obs=None, diverse_inst=False, model_awe=None):
    """
    Run the actual rollout on one subtask (which is one natural language instruction).
    """
    planned_actions = []
    if debug:
        print(f"{subtask} ", end="")
        time.sleep(0.5)
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
    model.reset()
    if model_awe != None:
        model_awe.reset()
    start_info = env.get_info()

    debug = False
    if debug:
        img_queue = []

    for step in range(EP_LEN):
        if model.replan != -1 and step % model.replan == 0:
            if model.model.module.refresh != -1:
                model.model.module.lang_encoder.lm_head.hidden_state = None
                model.model.module.lang_encoder.lm_head.history_memory = model.model.module.lang_encoder.lm_head.history_memory[
                                                                         -model.refresh:]
                if model_awe != None:
                    model_awe.model.module.lang_encoder.lm_head.hidden_state = None
                    model_awe.model.module.lang_encoder.lm_head.history_memory = model_awe.model.module.lang_encoder.lm_head.history_memory[
                                                                                 -model_awe.refresh:]
            else:
                model.reset()
                if model_awe != None:
                    model_awe.reset()
        # # 先获取waypoints
        # if model_awe != None:
        #     waypoints = model_awe.step(obs, lang_annotation, True)
        #     if len(planned_actions) == 0:
        #         waypoints_xyz = np.sum((waypoints ** 2)[0:3])
        #         # print(waypoints_xyz)
        #         if waypoints_xyz < 0.033:
        #             model.model.module.action_num = 1
        #         elif waypoints_xyz < 0.094:
        #             model.model.module.action_num = 2
        #         elif waypoints_xyz < 0.21:
        #             model.model.module.action_num = 3
        #         elif waypoints_xyz < 0.51:
        #             model.model.module.action_num = 4
        #         else:
        #             model.model.module.action_num = 5

        action = model.step(obs, lang_annotation, len(planned_actions) == 0)

        if len(planned_actions) == 0:
            if action.shape == (7,):
                planned_actions.append(action)
            else:
                planned_actions.extend([action[i] for i in range(action.shape[0])])
        action = planned_actions.pop(0)
        if model.use_diff:
            model.action_hist_queue.append(action)
        obs, _, _, current_info = env.step(action)
        if debug:
            img_copy = copy.deepcopy(obs['rgb_obs']['rgb_static'])
            img_queue.append(img_copy)
        if step == 0:
            # for tsne plot, only if available
            collect_plan(model, plans, subtask)

        # check if current step solves a task
        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
        if len(current_task_info) > 0:
            if debug:
                print(colored("success", "green"), end=" ")
                img_clip = ImageSequenceClip(img_queue, fps=30)
                img_clip.write_gif(os.path.join(eval_log_dir, f'{sequence_i}-{subtask_i}-{subtask}-succ.gif'), fps=30)
            return True
    if debug:
        print(colored("fail", "red"), end=" ")
        img_clip = ImageSequenceClip(img_queue, fps=30)
        img_clip.write_gif(os.path.join(eval_log_dir, f'{sequence_i}-{subtask_i}-{subtask}-fail.gif'), fps=30)
    return False


def eval_one_epoch_calvin(args, model, dataset_path, image_processor, tokenizer, future_act_len=-1):
    env = make_env(dataset_path)
    cast_dtype = get_cast_dtype(args.precision)
    wrapped_model = ModelWrapper(model, tokenizer, image_processor, cast_dtype, args.head_type == "diffusion",
                                 history_len=args.n_obs_steps, future_act_len=future_act_len)
    evaluate_policy(wrapped_model, env, 0, args.calvin_conf_path)


def eval_one_epoch_calvin_ddp(args, model, dataset_path, image_processor, tokenizer, eval_log_dir=None, debug=False,
                              future_act_len=-1, reset=False, diverse_inst=False, model_awe=None):
    env = make_env(dataset_path)
    cast_dtype = get_cast_dtype(args.precision)
    hist_len = None
    if args.head_type == "diffusion":
        hist_len = args.n_obs_steps
    elif args.pad_length != -1:
        hist_len = args.pad_length
    wrapped_model = ModelWrapper(model, tokenizer, image_processor, cast_dtype, args.head_type == "diffusion",
                                 history_len=hist_len, future_act_len=future_act_len)
    if model_awe != None:
        wrapped_model_awe = ModelWrapper(model_awe, tokenizer, image_processor, cast_dtype,
                                         args.head_type == "diffusion",
                                         history_len=hist_len, future_act_len=future_act_len)
    else:
        wrapped_model_awe = None

    evaluate_policy_ddp(wrapped_model, env, 0, args.calvin_conf_path, eval_log_dir=eval_log_dir, debug=debug,
                        reset=reset, diverse_inst=diverse_inst, model_awe=wrapped_model_awe)
    print('Model Average Step:')
    print(wrapped_model.avg_step)


def eval_one_epoch_calvin_ddp_awe(args, model, model_awe, dataset_path, image_processor, tokenizer, eval_log_dir=None,
                                  debug=False,
                                  future_act_len=-1, reset=False, diverse_inst=False):
    env = make_env(dataset_path)
    cast_dtype = get_cast_dtype(args.precision)
    hist_len = None
    if args.head_type == "diffusion":
        hist_len = args.n_obs_steps
    elif args.pad_length != -1:
        hist_len = args.pad_length
    wrapped_model = ModelWrapper(model, tokenizer, image_processor, cast_dtype, args.head_type == "diffusion",
                                 history_len=hist_len, future_act_len=future_act_len)
    wrapped_model_awe = ModelWrapper(model_awe, tokenizer, image_processor, cast_dtype, args.head_type == "diffusion",
                                     history_len=hist_len, future_act_len=future_act_len)
    evaluate_policy_ddp(wrapped_model, wrapped_model_awe, env, 0, args.calvin_conf_path, eval_log_dir=eval_log_dir,
                        debug=debug,
                        reset=reset, diverse_inst=diverse_inst)


def eval_one_epoch_calvin_with_dataloder(args, model, calvin_loader, image_processor, tokenizer, eval_log_dir=None,
                                         debug=False, future_act_len=-1, reset=False, diverse_inst=False):
    total_training_steps = calvin_loader.num_batches
    t = tqdm(
        enumerate(calvin_loader),
        disable=args.rank != 0,
        total=total_training_steps,
        initial=0,
    )
    cast_dtype = get_cast_dtype(args.precision)
    hist_len = hist_len = args.n_obs_steps if args.head_type == "diffusion" else args.pad_length if args.pad_length != -1 else None
    wrapped_model = ModelWrapper(model, tokenizer, image_processor, cast_dtype, args.head_type == "diffusion",
                                 history_len=hist_len, future_act_len=future_act_len)

    plans = defaultdict(list)
    results = []
    for num_steps, batch_calvin in t:
        assert len(batch_calvin) == 1
        batch_calvin = batch_calvin[0]  # 先假设只有一个batch
        if wrapped_model.model.module.env is None:
            dataset_path = batch_calvin['dataset_path']
            wrapped_model.model.module.env = get_env(Path(dataset_path), show_gui=False)  # make_env(dataset_path)
        batch_calvin.update({"env": wrapped_model.model.module.env})
        generator = batch_calvin.pop('generator')(**batch_calvin)

        wrapped_model.reset()

        debug = True
        img_queue = []
        planned_actions = []
        dt = next(generator)
        while not dt['done']:
            if debug:
                img_copy = copy.deepcopy(dt['rgb_static_ori'])
                img_queue.append(img_copy)

            step = dt["step_cur"]
            if wrapped_model.replan != -1 and step % wrapped_model.replan == 0:
                if wrapped_model.model.module.refresh != -1:
                    wrapped_model.model.module.lang_encoder.lm_head.hidden_state = None
                    wrapped_model.model.module.lang_encoder.lm_head.history_memory = wrapped_model.model.module.lang_encoder.lm_head.history_memory[
                                                                                     -wrapped_model.refresh:]
                else:
                    wrapped_model.reset()
                if dt["is_reset"]:
                    wrapped_model.reset()

            lang_annotation = dt["lang"]
            obs = {
                "rgb_obs": {
                    "rgb_static": dt['rgb_static_ori'],
                    "rgb_gripper": dt['rgb_gripper_ori'],
                },
                "robot_obs": dt['robot_obs'],
            }

            action = wrapped_model.step(obs, lang_annotation, (len(planned_actions) == 0))
            if len(planned_actions) == 0:
                if action.shape == (7,):
                    planned_actions.append(action)
                else:
                    planned_actions.extend([action[i] for i in range(action.shape[0])])

            action = planned_actions.pop(0)

            if wrapped_model.use_diff:
                wrapped_model.action_hist_queue.append(action)
            if step == 0:
                # for tsne plot, only if available
                subtask = dt["eval_sequence"][dt["subtask_i"]]
                collect_plan(wrapped_model, plans, subtask)

            dt = generator.send(action)

        success_counter = dt['success_counter']
        if debug:
            # print(colored("success", "green"), end=" ")
            img_clip = ImageSequenceClip(img_queue, fps=30)
            task_seq = f','.join(dt['eval_sequence'])
            img_clip.write_gif(os.path.join(eval_log_dir, f'{success_counter}-{task_seq}.gif'), fps=30)
            logging.info(os.path.join(eval_log_dir, f'{success_counter}-{task_seq}.gif'))

        results.extend([success_counter])

    def is_dist_avail_and_initialized():
        if not dist.is_available():
            return False
        if not dist.is_initialized():
            return False
        return True

    if is_dist_avail_and_initialized():
        dist.barrier()

    def merge_multi_list(res):
        tmp = []
        for l in res:
            tmp.extend(l)
        return tmp

    device_num = int(torch.distributed.get_world_size())
    all_res_tup = [copy.deepcopy(results) for _ in range(device_num)] if torch.distributed.get_rank() == 0 else None
    torch.distributed.gather_object(results, all_res_tup, dst=0)

    if torch.distributed.get_rank() == 0:
        results = merge_multi_list(all_res_tup)

    print(results)
    return results


def main():
    seed_everything(0, workers=True)  # type:ignore
    parser = argparse.ArgumentParser(description="Evaluate a trained model on multistep sequences with language goals.")
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset root directory.")

    # arguments for loading default model
    parser.add_argument(
        "--train_folder", type=str, help="If calvin_agent was used to train, specify path to the log dir."
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        default=None,
        help="Comma separated list of epochs for which checkpoints will be loaded",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path of the checkpoint",
    )
    parser.add_argument(
        "--last_k_checkpoints",
        type=int,
        help="Specify the number of checkpoints you want to evaluate (starting from last). Only used for calvin_agent.",
    )

    # arguments for loading custom model or custom language embeddings
    parser.add_argument(
        "--custom_model", action="store_true", help="Use this option to evaluate a custom model architecture."
    )

    parser.add_argument("--debug", action="store_true", help="Print debug info and visualize environment.")

    parser.add_argument("--eval_log_dir", default=None, type=str, help="Where to log the evaluation results.")

    parser.add_argument("--device", default=0, type=int, help="CUDA device")
    args = parser.parse_args()

    # evaluate a custom model
    if args.custom_model:
        model = CustomModel()
        env = make_env(args.dataset_path)
        evaluate_policy(model, env, debug=args.debug)
    else:
        assert "train_folder" in args

        checkpoints = []
        if args.checkpoints is None and args.last_k_checkpoints is None and args.checkpoint is None:
            print("Evaluating model with last checkpoint.")
            checkpoints = [get_last_checkpoint(Path(args.train_folder))]
        elif args.checkpoints is not None:
            print(f"Evaluating model with checkpoints {args.checkpoints}.")
            checkpoints = get_checkpoints_for_epochs(Path(args.train_folder), args.checkpoints)
        elif args.checkpoints is None and args.last_k_checkpoints is not None:
            print(f"Evaluating model with last {args.last_k_checkpoints} checkpoints.")
            checkpoints = get_all_checkpoints(Path(args.train_folder))[-args.last_k_checkpoints:]
        elif args.checkpoint is not None:
            checkpoints = [Path(args.checkpoint)]

        env = None
        for checkpoint in checkpoints:
            epoch = checkpoint.stem.split("=")[1]
            model, env, _ = get_default_model_and_env(
                args.train_folder,
                args.dataset_path,
                checkpoint,
                env=env,
                device_id=args.device,
            )
            evaluate_policy(model, env, epoch, eval_log_dir=args.eval_log_dir, debug=args.debug, create_plan_tsne=True)


def generate_zero_shot_instr():
    random.seed(123)
    with open(
            '/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/yanfeng/project/robotic/RoboFlamingo2/enrich_lang_annotations.json',
            'r') as f:
        val_annotations = json.load(f)
    eval_sequences = get_sequences(NUM_SEQUENCES)

    all_res = []
    for initial_state, eval_sequence in eval_sequences:
        res = []
        for subtask_i, subtask in enumerate(eval_sequence):
            res.append(random.choice(val_annotations[subtask]))
        all_res.append(res)
    with open(
            '/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/yanfeng/project/robotic/RoboFlamingo2/lang_annotation_cache.json',
            'w') as f:
        json.dump(all_res, f, indent=1)


def save_sequences():
    random.seed(123)
    eval_sequences = get_sequences(NUM_SEQUENCES)
    with open(
            '/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/yanfeng/project/robotic/RoboFlamingo2/eval_sequences.json',
            'w') as f:
        json.dump(eval_sequences, f)


def normalise_quat(x: torch.Tensor):
    return x / torch.clamp(x.square().sum(dim=-1).sqrt().unsqueeze(-1), min=1e-10)


def get_gripper_loc_bounds(buffer: float = 0.0, task: Optional[str] = None):
    path = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/huangyiyang02/new/3d_diffuser_actor/tasks/18_peract_tasks_traj_location_bounds.json"
    gripper_loc_bounds = json.load(open(path, "r"))
    if task is not None and task in gripper_loc_bounds:
        gripper_loc_bounds = gripper_loc_bounds[task]
        gripper_loc_bounds_min = np.array(gripper_loc_bounds[0]) - buffer
        gripper_loc_bounds_max = np.array(gripper_loc_bounds[1]) + buffer
        gripper_loc_bounds = np.stack([gripper_loc_bounds_min, gripper_loc_bounds_max])
    else:
        # Gripper workspace is the union of workspaces for all tasks
        gripper_loc_bounds = json.load(open(path, "r"))
        gripper_loc_bounds_min = np.min(np.stack([bounds[0] for bounds in gripper_loc_bounds.values()]),
                                        axis=0) - buffer
        gripper_loc_bounds_max = np.max(np.stack([bounds[1] for bounds in gripper_loc_bounds.values()]),
                                        axis=0) + buffer
        gripper_loc_bounds = np.stack([gripper_loc_bounds_min, gripper_loc_bounds_max])
    print("Gripper workspace size:", gripper_loc_bounds_max - gripper_loc_bounds_min)
    return gripper_loc_bounds


def unnormalize_pos(pos, gripper_loc_bounds):
    pos_min = gripper_loc_bounds[0].float().to(pos.device)
    pos_max = gripper_loc_bounds[1].float().to(pos.device)
    return (pos + 1.0) / 2.0 * (pos_max - pos_min) + pos_min


def get_waypoint_index(action,threshold):
    ith = 0
    while ith < len(action):
        action = action.cpu()
        action[..., :6] = np.cumsum(action[..., :6], axis=0)
        action[..., 6] = action[..., 6]

        jth = 1
        local_max_A = []  # 注意这里边只有一个
        p_st = action[0, :3]
        while jth < len(action):
            p_ed = action[jth, :3]
            distance_max = 0
            for kth in range(1, jth):
                p = action[kth, :3]
                if np.degrees(np.arccos(np.dot(p_ed - p, p_ed - p_st) / (
                        np.linalg.norm(p_ed - p) * np.linalg.norm(p_ed - p_st)))) > 90 or np.degrees(np.arccos(
                    np.dot(p_st - p, p_st - p_ed) / (np.linalg.norm(p_st - p) * np.linalg.norm(
                        p_st - p_ed)))) > 90:  # np.degrees(np.arccos(np.dot(p-p_st, p-p_ed) / (np.linalg.norm(p-p_st) * np.linalg.norm( p-p_ed))))
                    distance = 10000
                else:
                    distance = np.sin(np.arccos(np.dot(p - p_st, p_ed - p_st) / (
                            np.linalg.norm(p - p_st) * np.linalg.norm(p_ed - p_st)))) * np.linalg.norm(p - p_st)
                distance_max = max(distance_max, distance)
            # print(distance_max)
            if distance_max > threshold:  # 0.03:
                local_max_A.append(jth - 1)
                break
            jth += 1

        def gripper_state_changed(trajectories):
            trajectories = np.vstack([trajectories[:1], trajectories])
            openess = trajectories[:, -1]
            changed = openess[:-1] != openess[1:]
            return np.where(changed)[0]

        # waypoints are frames with changing gripper states
        gripper_changed = gripper_state_changed(action)
        one_frame_before_gripper_changed = (
                gripper_changed[gripper_changed > 1] - 1  # 第0个是不加入的
        )
        # waypoints is the last pose in the trajectory
        last_frame = [len(action) - 1]

        keyframe_inds = (
                last_frame +
                gripper_changed.tolist() +
                one_frame_before_gripper_changed.tolist() +
                local_max_A
        )
        keyframe_inds = np.unique(keyframe_inds)
        keyframe_inds.sort()
        assert keyframe_inds[0] != 0
        ith += keyframe_inds[0]
        return ith  # 表示该走几步

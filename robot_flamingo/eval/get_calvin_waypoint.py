import numpy as np
from pathlib import Path
import sys
import copy
from scipy.spatial.transform import Rotation
sys.path.append("/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/huangyiyang02/new/awe")
sys.path.append('/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/huangyiyang02/new/awe/robosuite')
import typing
import math
from tqdm import tqdm
import json
import matplotlib as mpl
import matplotlib.pyplot as plt

def plot_3d_trajectory(ax, traj_list, label, gripper=None, legend=True, add=None):
    """Plot a 3D trajectory."""
    l = label
    num_frames = len(traj_list)
    for i in range(num_frames):
        # change the color if the gripper state changes
        gripper_state_changed = (
            gripper is not None and i > 0 and gripper[i] != gripper[i - 1]
        )
        if label == "pred" or label == "waypoints":
            if gripper_state_changed or (add is not None and i in add):
                c = mpl.cm.Oranges(0.2 + 0.5 * i / num_frames)
            else:
                c = mpl.cm.Reds(0.5 + 0.5 * i / num_frames)
        elif label == "gt" or label == "ground truth":
            if gripper_state_changed:
                c = mpl.cm.Greens(0.2 + 0.5 * i / num_frames)
            else:
                c = mpl.cm.Blues(0.5 + 0.5 * i / num_frames)
        else:
            c = mpl.cm.Greens(0.5 + 0.5 * i / num_frames)

        # change the marker if the gripper state changes
        if gripper_state_changed:
            if gripper[i] == 1:  # open
                marker = "D"
            else:  # close
                marker = "s"
        else:
            marker = "o"

        # plot the vector between the current and the previous point
        if (label == "pred" or label == "action" or label == "waypoints") and i > 0:
            v = traj_list[i] - traj_list[i - 1]
            ax.quiver(
                traj_list[i - 1][0],
                traj_list[i - 1][1],
                traj_list[i - 1][2],
                v[0],
                v[1],
                v[2],
                color="r",
                alpha=0.5,
                # linewidth=3,
            )

        # if label is waypoint, make the marker D, and slightly bigger
        if add is not None and i in add:
            marker = "D"
            ax.plot(
                [traj_list[i][0]],
                [traj_list[i][1]],
                [traj_list[i][2]],
                marker=marker,
                label=l,
                color=c,
                markersize=10,
            )
        else:
            ax.plot(
                [traj_list[i][0]],
                [traj_list[i][1]],
                [traj_list[i][2]],
                marker=marker,
                label=l,
                color=c,
                # markersize=10,
            )
        l = None

    if legend:
        ax.legend()

PI = np.pi
EPS = np.finfo(float).eps * 4.0


def load_npz(filename: Path):
    return np.load(filename.as_posix())

def total_traj_err(err_list):
    # return np.mean(err_list)
    return np.max(err_list)

def get_episode_name(file_idx: int):
    abs_datasets_dir = Path("/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/yanfeng/data/robotics/task_D_D/training")
    return Path(
        abs_datasets_dir / f"episode_{file_idx:0{7}d}.npz"
    )

def linear_interpolation(p1, p2, t):
    """Compute the linear interpolation between two 3D points"""
    return p1 + t * (p2 - p1)

def unit_vector(data, axis=None, out=None):
    """
    Returns ndarray normalized by length, i.e. eucledian norm, along axis.

    E.g.:
        >>> v0 = numpy.random.random(3)
        >>> v1 = unit_vector(v0)
        >>> numpy.allclose(v1, v0 / numpy.linalg.norm(v0))
        True

        >>> v0 = numpy.random.rand(5, 4, 3)
        >>> v1 = unit_vector(v0, axis=-1)
        >>> v2 = v0 / numpy.expand_dims(numpy.sqrt(numpy.sum(v0*v0, axis=2)), 2)
        >>> numpy.allclose(v1, v2)
        True

        >>> v1 = unit_vector(v0, axis=1)
        >>> v2 = v0 / numpy.expand_dims(numpy.sqrt(numpy.sum(v0*v0, axis=1)), 1)
        >>> numpy.allclose(v1, v2)
        True

        >>> v1 = numpy.empty((5, 4, 3), dtype=numpy.float32)
        >>> unit_vector(v0, axis=1, out=v1)
        >>> numpy.allclose(v1, v2)
        True

        >>> list(unit_vector([]))
        []

        >>> list(unit_vector([1.0]))
        [1.0]

    Args:
        data (np.array): data to normalize
        axis (None or int): If specified, determines specific axis along data to normalize
        out (None or np.array): If specified, will store computation in this variable

    Returns:
        None or np.array: If @out is not specified, will return normalized vector. Otherwise, stores the output in @out
    """
    if out is None:
        data = np.array(data, dtype=np.float32, copy=True)
        if data.ndim == 1:
            data /= math.sqrt(np.dot(data, data))
            return data
    else:
        if out is not data:
            out[:] = np.array(data, copy=False)
        data = out
    length = np.atleast_1d(np.sum(data * data, axis))
    np.sqrt(length, length)
    if axis is not None:
        length = np.expand_dims(length, axis)
    data /= length
    if out is None:
        return data

def quat_slerp(quat0, quat1, fraction, shortestpath=True):
    """
    Return spherical linear interpolation between two quaternions.

    E.g.:
    >>> q0 = random_quat()
    >>> q1 = random_quat()
    >>> q = quat_slerp(q0, q1, 0.0)
    >>> np.allclose(q, q0)
    True

    >>> q = quat_slerp(q0, q1, 1.0)
    >>> np.allclose(q, q1)
    True

    >>> q = quat_slerp(q0, q1, 0.5)
    >>> angle = math.acos(np.dot(q0, q))
    >>> np.allclose(2.0, math.acos(np.dot(q0, q1)) / angle) or \
        np.allclose(2.0, math.acos(-np.dot(q0, q1)) / angle)
    True

    Args:
        quat0 (np.array): (x,y,z,w) quaternion startpoint
        quat1 (np.array): (x,y,z,w) quaternion endpoint
        fraction (float): fraction of interpolation to calculate
        shortestpath (bool): If True, will calculate the shortest path

    Returns:
        np.array: (x,y,z,w) quaternion distance
    """
    q0 = unit_vector(quat0[:4])
    q1 = unit_vector(quat1[:4])
    if fraction == 0.0:
        return q0
    elif fraction == 1.0:
        return q1
    d = np.dot(q0, q1)
    if abs(abs(d) - 1.0) < EPS:
        return q0
    if shortestpath and d < 0.0:
        # invert rotation
        d = -d
        q1 *= -1.0
    angle = math.acos(np.clip(d, -1, 1))
    if abs(angle) < EPS:
        return q0
    isin = 1.0 / math.sin(angle)
    q0 *= math.sin((1.0 - fraction) * angle) * isin
    q1 *= math.sin(fraction * angle) * isin
    q0 += q1
    return q0

def point_quat_distance(point, quat_start, quat_end, t, total):
    pred_point = quat_slerp(quat_start, quat_end, fraction=t / total)
    err_quat = (
        Rotation.from_quat(pred_point) * Rotation.from_quat(point).inv()
    ).magnitude()
    return err_quat

def point_line_distance(point, line_start, line_end):
    """Compute the shortest distance between a 3D point and a line segment defined by two 3D points"""
    line_vector = line_end - line_start
    point_vector = point - line_start
    # t represents the position of the orthogonal projection of the given point onto the infinite line defined by the segment
    t = np.dot(point_vector, line_vector) / np.dot(line_vector, line_vector)
    t = np.clip(t, 0, 1)
    projection = linear_interpolation(line_start, line_end, t)
    return np.linalg.norm(point - projection)


def pos_only_geometric_waypoint_trajectory(
    actions, gt_states, waypoints, return_list=False
):
    """Compute the geometric trajectory from the waypoints"""

    # prepend 0 to the waypoints for geometric computation
    if waypoints[0] != 0:
        waypoints = [0] + waypoints

    keypoints_pos = [actions[k] for k in waypoints]
    state_err = []
    n_segments = len(waypoints) - 1

    for i in range(n_segments):
        # Get the current keypoint and the next keypoint
        start_keypoint_pos = keypoints_pos[i]
        end_keypoint_pos = keypoints_pos[i + 1]

        # Select ground truth points within the current segment
        start_idx = waypoints[i]
        end_idx = waypoints[i + 1]
        segment_points_pos = gt_states[start_idx:end_idx]

        # Compute the shortest distances between ground truth points and the current segment
        for i in range(len(segment_points_pos)):
            pos_err = point_line_distance(
                segment_points_pos[i], start_keypoint_pos, end_keypoint_pos
            )
            state_err.append(pos_err)

    # print the average and max error
    # print(
    #     f"Average pos error: {np.mean(state_err):.6f} \t Max pos error: {np.max(state_err):.6f}"
    # )

    if return_list:
        return total_traj_err(state_err), state_err
    else:
        return total_traj_err(state_err)
def geometric_waypoint_trajectory(actions, gt_states, waypoints, return_list=False):
    """Compute the geometric trajectory from the waypoints"""

    # prepend 0 to the waypoints for geometric computation
    if waypoints[0] != 0:
        waypoints = [0] + waypoints
    gt_pos = [p["robot0_eef_pos"] for p in gt_states]
    gt_quat = [p["robot0_eef_quat"] for p in gt_states]

    keypoints_pos = [actions[k, :3] for k in waypoints]
    keypoints_quat = [gt_quat[k] for k in waypoints]

    state_err = []

    n_segments = len(waypoints) - 1

    for i in range(n_segments):
        # Get the current keypoint and the next keypoint
        start_keypoint_pos = keypoints_pos[i]
        end_keypoint_pos = keypoints_pos[i + 1]
        start_keypoint_quat = keypoints_quat[i]
        end_keypoint_quat = keypoints_quat[i + 1]

        # Select ground truth points within the current segment
        start_idx = waypoints[i]
        end_idx = waypoints[i + 1]
        segment_points_pos = gt_pos[start_idx:end_idx]
        segment_points_quat = gt_quat[start_idx:end_idx]

        # Compute the shortest distances between ground truth points and the current segment
        for i in range(len(segment_points_pos)):
            pos_err = point_line_distance(
                segment_points_pos[i], start_keypoint_pos, end_keypoint_pos
            )
            rot_err = point_quat_distance(
                segment_points_quat[i],
                start_keypoint_quat,
                end_keypoint_quat,
                i,
                len(segment_points_quat),
            )
            state_err.append(pos_err + rot_err)

    # print the average and max error for pos and rot
    # print(f"Average pos error: {np.mean(pos_err_list):.6f} \t Average rot error: {np.mean(rot_err_list):.6f}")
    # print(f"Max pos error: {np.max(pos_err_list):.6f} \t Max rot error: {np.max(rot_err_list):.6f}")

    if return_list:
        return total_traj_err(state_err), state_err
    return total_traj_err(state_err)

def dp_waypoint_selection(
        env=None,
        actions=None,
        gt_states=None,
        err_threshold=None,
        initial_states=None,
        remove_obj=None,
        pos_only=False,
):
    if actions is None:
        actions = copy.deepcopy(gt_states)
    elif gt_states is None:
        gt_states = copy.deepcopy(actions)

    num_frames = len(actions)

    # make the last frame a waypoint
    initial_waypoints = [num_frames - 1]

    # make the frames of gripper open/close waypoints
    if not pos_only:
        for i in range(num_frames - 1):
            if actions[i, -1] != actions[i + 1, -1]:
                initial_waypoints.append(i)
                # initial_waypoints.append(i + 1)
        initial_waypoints.sort()

    # Memoization table to store the waypoint sets for subproblems
    memo = {}

    # Initialize the memoization table
    for i in range(num_frames):
        memo[i] = (0, [])

    memo[1] = (1, [1])
    func = pos_only_geometric_waypoint_trajectory

    # Check if err_threshold is too small, then return all points as waypoints
    min_error = func(actions, gt_states, list(range(1, num_frames)))
    if err_threshold < min_error:
        print("Error threshold is too small, returning all points as waypoints.")
        return list(range(1, num_frames))

    # Populate the memoization table using an iterative bottom-up approach
    for i in range(1, num_frames):
        min_waypoints_required = float("inf")
        best_waypoints = []

        for k in range(1, i):
            # waypoints are relative to the subsequence
            waypoints = [j - k for j in initial_waypoints if j >= k and j < i] + [i - k]

            total_traj_err = func(
                actions=actions[k: i + 1],
                gt_states=gt_states[k: i + 1],
                waypoints=waypoints,
            )

            if total_traj_err < err_threshold:
                subproblem_waypoints_count, subproblem_waypoints = memo[k - 1]
                total_waypoints_count = 1 + subproblem_waypoints_count

                if total_waypoints_count < min_waypoints_required:
                    min_waypoints_required = total_waypoints_count
                    best_waypoints = subproblem_waypoints + [i]

        memo[i] = (min_waypoints_required, best_waypoints)

    min_waypoints_count, waypoints = memo[num_frames - 1]
    waypoints += initial_waypoints
    # remove duplicates
    waypoints = list(set(waypoints))
    waypoints.sort()
    # print(
    #     f"Minimum number of waypoints: {len(waypoints)} \tTrajectory Error: {total_traj_err}"
    # )
    # print(f"waypoint positions: {waypoints}")

    return waypoints




if __name__ == "__main__":

    abs_datasets_dir = Path("/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/yanfeng/data/robotics/task_D_D/training")
    lang_folder = "lang_annotations"
    lang_data = np.load(
        abs_datasets_dir / lang_folder / "auto_lang_ann.npy",
        allow_pickle=True,
    ).item()
    ep_start_end_ids = lang_data["info"]["indx"]  # each of them are 64
    episode_waypoint_list = {}
    for i,(start,end) in tqdm(enumerate(ep_start_end_ids),total=len(ep_start_end_ids)):
        episodes = [
            load_npz(get_episode_name(j))
            for j in range(start, end)
        ]
        key = "actions"
        key_robot = "robot_obs"
        key_delta = "rel_actions"
        episode = {key: np.stack([ep[key] for ep in episodes])}
        robot = {key_robot: np.stack([ep[key_robot] for ep in episodes])}
        dt = {key_delta: np.stack([ep[key_delta] for ep in episodes])}
        rel_episode = episode["actions"]
        rel_robot = robot["robot_obs"]
        rel_dt = dt["rel_actions"]
        waypoints = dp_waypoint_selection(
            actions=rel_episode,
            err_threshold=0.004
        )
        # left_arm_xyz = rel_episode[:, :3]
        # right_arm_xyz = rel_episode[:, :3]
        #
        # # Find global min and max for each axis
        # all_data = np.concatenate([left_arm_xyz, right_arm_xyz], axis=0)
        # min_x, min_y, min_z = np.min(all_data, axis=0)
        # max_x, max_y, max_z = np.max(all_data, axis=0)
        #
        # fig = plt.figure(figsize=(20, 10))
        # ax1 = fig.add_subplot(121, projection="3d")
        # ax1.set_xlabel("x")
        # ax1.set_ylabel("y")
        # ax1.set_zlabel("z")
        # ax1.set_title("Left", fontsize=20)
        # ax1.set_xlim([min_x, max_x])
        # ax1.set_ylim([min_y, max_y])
        # ax1.set_zlim([min_z, max_z])
        #
        # plot_3d_trajectory(ax1, left_arm_xyz, label="ground truth", legend=False)
        #
        # ax2 = fig.add_subplot(122, projection="3d")
        # ax2.set_xlabel("x")
        # ax2.set_ylabel("y")
        # ax2.set_zlabel("z")
        # ax2.set_title("Right", fontsize=20)
        # ax2.set_xlim([min_x, max_x])
        # ax2.set_ylim([min_y, max_y])
        # ax2.set_zlim([min_z, max_z])
        #
        # plot_3d_trajectory(ax2, right_arm_xyz, label="ground truth", legend=False)
        #
        # # prepend 0 to waypoints to include the initial state
        # waypoints = [0] + waypoints
        #
        # plot_3d_trajectory(
        #     ax1,
        #     [left_arm_xyz[i] for i in waypoints],
        #     label="waypoints",
        #     legend=False,
        # )  # Plot waypoints for left_arm_xyz
        # plot_3d_trajectory(
        #     ax2,
        #     [right_arm_xyz[i] for i in waypoints],
        #     label="waypoints",
        #     legend=False,
        # )  # Plot waypoints for right_arm_xyz
        #
        # fig.suptitle(f"Task:", fontsize=30)
        #
        # handles, labels = ax1.get_legend_handles_labels()
        # fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=20)
        #
        # # fig.savefig(
        # #     "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/huangyiyang02/new/temp/waypoints2.png"
        # # )
        # # plt.close(fig)
        # plt.show()

        j = 0
        for i in range(len(rel_episode)):
            if i == 0:
                episode_waypoint_list[str(i + start)] = rel_dt[i].tolist()
                continue
            if i > waypoints[j]:
                j = j + 1
            delta_action = np.zeros(7,dtype=float)
            delta_action[0:3] = np.clip((rel_episode[waypoints[j]][0:3] - rel_robot[i][0:3]) * 50,-1,1)
            delta_action[3:6] = np.clip((rel_episode[waypoints[j]][3:6] - rel_robot[i][3:6]) * 20,-1,1)
            delta_action[6] = rel_episode[waypoints[j]][6]
            episode_waypoint_list[str(i + start)] = delta_action.tolist()

    file_path = '/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/yanfeng/data/robotics/task_D_D/waypoints_delta.json'

    print(len(episode_waypoint_list.keys()))

    # 将字典存储为JSON文件
    with open(file_path, 'w') as file:
        json.dump(episode_waypoint_list, file)



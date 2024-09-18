#%%
import os
import h5py
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
from

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


def main(args):
    num_waypoints = []
    num_frames = []

    # load data
    for i in tqdm(range(args.start_idx, args.end_idx + 1)):
        dataset_path = os.path.join(args.dataset, f"episode_{i}.hdf5")
        with h5py.File(dataset_path, "r+") as root:
            qpos = root["/observations/qpos"][()]

            if args.use_ee:
                qpos = np.array(qpos)  # ts, dim

                # calculate EE pose
                from act.convert_ee import get_ee

                left_arm_ee = get_ee(qpos[:, :6], qpos[:, 6:7])
                right_arm_ee = get_ee(qpos[:, 7:13], qpos[:, 13:14])
                qpos = np.concatenate([left_arm_ee, right_arm_ee], axis=1)

            # select waypoints
            waypoints = dp_waypoint_selection( # if it's too slow, use greedy_waypoint_selection
                env=None,
                actions=qpos,
                gt_states=qpos,
                err_threshold=args.err_threshold,
                pos_only=True,
            )
            print(
                f"Episode {i}: {len(qpos)} frames -> {len(waypoints)} waypoints (ratio: {len(qpos)/len(waypoints):.2f})"
            )
            num_waypoints.append(len(waypoints))
            num_frames.append(len(qpos))

            # save waypoints
            if args.save_waypoints:
                name = f"/waypoints"
                try:
                    root[name] = waypoints
                except:
                    # if the waypoints dataset already exists, ask the user if they want to overwrite
                    print("waypoints dataset already exists. Overwrite? (y/n)")
                    ans = input()
                    if ans == "y":
                        del root[name]
                        root[name] = waypoints

            # visualize ground truth qpos and waypoints
            if args.plot_3d:
                if not args.use_ee:
                    qpos = np.array(qpos)  # ts, dim
                    from act.convert_ee import get_xyz

                    left_arm_xyz = get_xyz(qpos[:, :6])
                    right_arm_xyz = get_xyz(qpos[:, 7:13])
                else:
                    left_arm_xyz = left_arm_ee[:, :3]
                    right_arm_xyz = right_arm_ee[:, :3]

                # Find global min and max for each axis
                all_data = np.concatenate([left_arm_xyz, right_arm_xyz], axis=0)
                min_x, min_y, min_z = np.min(all_data, axis=0)
                max_x, max_y, max_z = np.max(all_data, axis=0)

                fig = plt.figure(figsize=(20, 10))
                ax1 = fig.add_subplot(121, projection="3d")
                ax1.set_xlabel("x")
                ax1.set_ylabel("y")
                ax1.set_zlabel("z")
                ax1.set_title("Left", fontsize=20)
                ax1.set_xlim([min_x, max_x])
                ax1.set_ylim([min_y, max_y])
                ax1.set_zlim([min_z, max_z])

                plot_3d_trajectory(ax1, left_arm_xyz, label="ground truth", legend=False)

                ax2 = fig.add_subplot(122, projection="3d")
                ax2.set_xlabel("x")
                ax2.set_ylabel("y")
                ax2.set_zlabel("z")
                ax2.set_title("Right", fontsize=20)
                ax2.set_xlim([min_x, max_x])
                ax2.set_ylim([min_y, max_y])
                ax2.set_zlim([min_z, max_z])

                plot_3d_trajectory(ax2, right_arm_xyz, label="ground truth", legend=False)

                # prepend 0 to waypoints to include the initial state
                waypoints = [0] + waypoints

                plot_3d_trajectory(
                    ax1,
                    [left_arm_xyz[i] for i in waypoints],
                    label="waypoints",
                    legend=False,
                )  # Plot waypoints for left_arm_xyz
                plot_3d_trajectory(
                    ax2,
                    [right_arm_xyz[i] for i in waypoints],
                    label="waypoints",
                    legend=False,
                )  # Plot waypoints for right_arm_xyz

                fig.suptitle(f"Task: {args.dataset.split('/')[-1]}", fontsize=30)

                handles, labels = ax1.get_legend_handles_labels()
                fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=20)

                fig.savefig(
                    f"plot/act/{args.dataset.split('/')[-1]}_{i}_t_{args.err_threshold}_waypoints.png"
                )
                plt.close(fig)

            root.close()

    print(
        f"Average number of waypoints: {np.mean(num_waypoints)} \tAverage number of frames: {np.mean(num_frames)} \tratio: {np.mean(num_frames)/np.mean(num_waypoints)}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/act/sim_transfer_cube_scripted",
        # default="data/act/sim_insertion_scripted",
        # default="data/act/sim_transfer_cube_human",
        # default="data/act/sim_insertion_human",
        # default="data/act/aloha_screw_driver",
        # default="data/act/aloha_coffee",
        # default="data/act/aloha_towel",
        # default="data/act/aloha_coffee_new",
        help="path to hdf5 dataset",
    )

    # index of the trajectory to playback. If omitted, playback trajectory 0.
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="(optional) start index of the trajectory to playback",
    )

    parser.add_argument(
        "--end_idx",
        type=int,
        default=49,
        help="(optional) end index of the trajectory to playback",
    )

    # error threshold for reconstructing the trajectory
    parser.add_argument(
        "--err_threshold",
        type=float,
        default=0.05,
        help="(optional) error threshold for reconstructing the trajectory",
    )

    # whether to save waypoints
    parser.add_argument(
        "--save_waypoints",
        action="store_true",
        help="(optional) whether to save waypoints",
    )

    # whether to use the ee space for waypoint selection
    parser.add_argument(
        "--use_ee",
        action="store_true",
        help="(optional) whether to use the ee space for waypoint selection",
    )

    # whether to plot 3d
    parser.add_argument(
        "--plot_3d",
        action="store_true",
        help="(optional) whether to plot 3d",
    )
#%%
    dataset_dir = Path("/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/yanfeng/data/robotics/task_D_D/training/")
    annotation_dir = os.path.join(dataset_dir, 'lang_annotations/auto_lang_ann.npy')
    lang_data = np.load(
        annotation_dir, allow_pickle=True
    ).item()
    print(lang_data["info"]["indx"])
    start_idx = lang_data["info"]["indx"][0][0]
    end_idx = lang_data["info"]["indx"][0][1]

    episode_array = np.zeros((end_idx - start_idx + 1,7))
    for i in tqdm(range(start_idx, end_idx + 1)):
        file_path = os.path.join(dataset_dir, f"episode_{i:07d}.npz")
        data_info = np.load(Path(file_path).as_posix())
        episode_array[[i - start_idx]] = data_info["actions"]



    # args = parser.parse_args()
    # main(args)

#%%
import torch
import numpy as np

# 创建一个具有多个维度的张量
tensor = torch.randn(1,1, 2, 3, 256, 256)
#%%
import torch
time_var = torch.arange(5).float()

time_matrix = torch.cat([time_var ** r for r in range(4)], dim=0).reshape(4, 5)
print(time_matrix)
time_matrix = time_matrix / time_matrix[..., -1:]  # 为了放大高次系数，除了最后一维
print(time_matrix)
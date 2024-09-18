import numpy as np

def get_waypoint_index(action):
    is_waypoint = np.zeros(len(action), dtype=np.bool_)
    action_sum = np.zeros(action.shape)
    ith = 0
    while ith < len(action):
        action_sum[...,:6] = np.cumsum(action[..., :6], axis=0)
        action_sum[...,6] = action[...,6]

        jth = 1
        local_max_A = []  # 注意这里边只有一个
        p_st = action[0,:3]
        while jth < len(action_sum):
            p_ed = action_sum[jth,:3]
            distance_max = 0
            for kth in range(1, jth):
                p = action[kth,:3]
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
            if distance_max > 0.03:  # 0.03:
                local_max_A.append(jth - 1)
                break
            jth += 1

        def gripper_state_changed(trajectories):
            trajectories = np.vstack([trajectories[:1], trajectories])
            openess = trajectories[:, -1]
            changed = openess[:-1] != openess[1:]
            return np.where(changed)[0]

        # waypoints are frames with changing gripper states
        gripper_changed = gripper_state_changed(action_sum)
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
        return ith + 1

action_example = np.random.rand(9,7)
get_waypoint_index(action_example)
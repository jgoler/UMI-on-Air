#!/usr/bin/env python3
"""Episode recorder that works over SSH with X11 forwarding.

Uses EGL for MuJoCo offscreen rendering and OpenCV for display + keyboard input.
No pynput or GLX dependencies.

Usage:
    MUJOCO_GL=egl python record_episodes_ssh.py --task umi_pick
"""
import os
# Force EGL rendering before any MuJoCo import
os.environ.setdefault("MUJOCO_GL", "egl")

import time
import h5py
import argparse
import numpy as np
import cv2

from constants import DT, SIM_TASK_CONFIGS
from ee_sim_env import make_ee_sim_env
from keyboard_policy_cv2 import CVKeyboardPolicy


def get_auto_index(dataset_dir):
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir, exist_ok=True)
    for i in range(1001):
        if not os.path.isfile(os.path.join(dataset_dir, f'episode_{i}.hdf5')):
            return i
    raise RuntimeError("More than 1000 episodes")


def save_episode(dataset_dir, camera_names, idx, episode, action_traj, env_init_state):
    if len(action_traj) == 0:
        print("[WARN] No timesteps to save.")
        return
    T = len(action_traj)
    qpos_arr = np.stack([episode[t].observation['qpos'] for t in range(T)])
    act_arr = np.stack(action_traj)
    img_arrays = {
        cam: np.stack([episode[t].observation['images'][cam] for t in range(T)])
        for cam in camera_names
    }
    tmp_path = os.path.join(dataset_dir, f'episode_{idx}.tmp')
    final_path = os.path.join(dataset_dir, f'episode_{idx}.hdf5')
    t0 = time.time()
    with h5py.File(tmp_path, 'w', rdcc_nbytes=1024**2 * 2) as root:
        root.attrs['sim'] = True
        root.attrs['success'] = True
        root.create_dataset('env_init_state', data=env_init_state)
        obs = root.create_group('observations')
        obs.create_dataset('qpos', data=qpos_arr)
        root.create_dataset('action', data=act_arr)
        img_grp = obs.create_group('images')
        for cam, arr in img_arrays.items():
            img_grp.create_dataset(cam, data=arr, dtype='uint8',
                                   chunks=(1,) + arr.shape[1:])
    os.replace(tmp_path, final_path)
    print(f"Saved {final_path} ({time.time() - t0:.1f}s)")


def main():
    parser = argparse.ArgumentParser(description="Record episodes via SSH/X11")
    parser.add_argument('--task', '--task_name', dest='task_name', required=True)
    parser.add_argument('--episode_idx', type=int, default=None)
    parser.add_argument('--disturb', action='store_true')
    parser.add_argument('--width', type=int, default=960, help='Display window width')
    parser.add_argument('--height', type=int, default=720, help='Display window height')
    args = parser.parse_args()

    task_name = args.task_name
    task_cfg = SIM_TASK_CONFIGS[task_name]
    dataset_dir = task_cfg['dataset_dir']
    episode_len = task_cfg['episode_len']
    camera_names = task_cfg['camera_names']
    render_cam = 'ee'

    episode_idx = args.episode_idx if args.episode_idx is not None else get_auto_index(dataset_dir)

    env = make_ee_sim_env(task_name, camera_names=camera_names,
                          disturbance_enabled=args.disturb)
    policy = CVKeyboardPolicy(task_name=task_name)

    steps_until_stop_success = int(3.0 / DT)
    win = 'UMI Simulation'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, args.width, args.height)

    disp_w, disp_h = args.width, args.height

    def show(img_rgb, overlay_text=None, color=(0, 255, 0)):
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        img_res = cv2.resize(img_bgr, (disp_w, disp_h))
        if overlay_text:
            cv2.putText(img_res, overlay_text, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        # Speed indicator
        spd = policy.current_speed_idx + 1
        cv2.putText(img_res, f"Speed: {spd}/3", (20, disp_h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 255), 1)
        cv2.imshow(win, img_res)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    while True:
        print(f"\n=== Episode {episode_idx} ===")
        ts = env.reset()
        env_init_state = env.physics.data.qpos.copy()
        policy.generate_trajectory(ts)
        episode = [ts]
        action_traj = []
        exit_program = False
        restart_episode = False

        # --- READY phase: preview + wait for P ---
        print("Press P to start recording | ESC to exit | R to reset scene")
        recording_started = False
        countdown_start = None

        while not recording_started:
            action = policy.get_action()
            ts = env.step(action)

            if countdown_start is not None:
                elapsed = time.time() - countdown_start
                remaining = 3 - int(elapsed)
                if elapsed >= 3.0:
                    recording_started = True
                    episode = [ts]
                    print("Recording!")
                    break
                show(ts.observation['images'][render_cam],
                     f"Starting in {max(1, remaining)}...", (0, 0, 255))
            else:
                show(ts.observation['images'][render_cam],
                     f"READY - EP {episode_idx} (P=record, R=reset)")

            key = cv2.waitKeyEx(20)  # 20ms = ~50Hz
            cmd = policy.process_key(key)
            if cmd == "exit":
                exit_program = True
                break
            elif cmd == "toggle_record" and countdown_start is None:
                countdown_start = time.time()
                print("Starting in 3...")
            elif cmd == "reset":
                print("Resetting scene...")
                ts = env.reset()
                policy.generate_trajectory(ts)
                episode = [ts]
            elif cmd == "delete_last":
                if episode_idx > 0:
                    last = os.path.join(dataset_dir, f'episode_{episode_idx - 1}.hdf5')
                    if os.path.isfile(last):
                        os.remove(last)
                        episode_idx -= 1
                        print(f"Deleted. Now at episode_{episode_idx}.")

        if exit_program:
            break

        # --- RECORDING phase ---
        step = 0
        success_detected = False
        success_counter = 0

        while step < episode_len:
            step_start = time.time()

            key = cv2.waitKeyEx(1)
            cmd = policy.process_key(key)
            if cmd == "exit":
                exit_program = True
                break
            elif cmd == "reset":
                print("Restart requested.")
                restart_episode = True
                break

            action_traj.append(policy.get_action())
            ts = env.step(action_traj[-1])
            episode.append(ts)

            if success_detected:
                remaining = max(0, (steps_until_stop_success - success_counter) * DT)
                show(ts.observation['images'][render_cam],
                     f"SUCCESS - stopping in {remaining:.1f}s", (0, 255, 0))
            else:
                show(ts.observation['images'][render_cam],
                     f"REC EP{episode_idx}: {step+1}/{episode_len}", (0, 0, 255))

            # Success detection
            if not success_detected and ts.reward == env.task.max_reward:
                success_detected = True
                success_counter = 0
                print("Success reached - finishing in 3s")

            if success_detected:
                success_counter += 1
                if success_counter >= steps_until_stop_success:
                    print("Success countdown done.")
                    break

            step += 1

            elapsed = time.time() - step_start
            if elapsed < DT:
                time.sleep(DT - elapsed)

        if exit_program:
            break

        # --- Save or discard ---
        success_episode = success_detected and success_counter >= steps_until_stop_success
        if restart_episode or not success_episode:
            if restart_episode:
                print("Episode restarted.")
            else:
                print("Episode did not reach success - discarding.")
            continue

        save_episode(dataset_dir, camera_names, episode_idx, episode[:-1], action_traj, env_init_state)
        episode_idx += 1

    policy.cleanup()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == '__main__':
    main()

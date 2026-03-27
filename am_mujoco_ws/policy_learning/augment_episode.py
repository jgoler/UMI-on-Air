"""
augment_episode.py  -  Milestone 1: one augmented umi_pick trajectory.

Steps
-----
1. Load a recorded episode.
2. Detect grasp time from gripper channel.
3. Apply a time-varying SE(3) perturbation that decays to identity by grasp time.
4. Replay augmented trajectory through MuJoCo oracle sim.
5. Save augmented HDF5 + visualisation plots.

Usage
-----
    python augment_episode.py
    python augment_episode.py --no_replay          # math + plot only, no sim
    python augment_episode.py --delta_pos 0.05 0.03 0.0 --delta_rot_deg 5 3 2
"""

import os
import sys
import argparse
import h5py
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation, Slerp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from constants import SIM_TASK_CONFIGS, DT
from ee_sim_env import make_ee_sim_env


# ── SE(3) helpers ─────────────────────────────────────────────────────────────

def quat_wxyz_to_scipy(q):
    """[qw,qx,qy,qz] → scipy Rotation."""
    return Rotation.from_quat([q[1], q[2], q[3], q[0]])

def scipy_to_quat_wxyz(r):
    """scipy Rotation → [qw,qx,qy,qz]."""
    xyzw = r.as_quat()
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])

def apply_se3(R_delta, t_delta, pos, quat_wxyz):
    """Apply SE(3) (R_delta, t_delta) to pose: p' = R@p + t, q' = R*q."""
    pos_new  = R_delta.apply(pos) + t_delta
    quat_new = scipy_to_quat_wxyz(R_delta * quat_wxyz_to_scipy(quat_wxyz))
    return pos_new, quat_new


# ── Augmentation ──────────────────────────────────────────────────────────────

def detect_grasp_time(qpos, close_thresh=0.6):
    """First timestep where gripper stays closed for 10 consecutive steps."""
    closed = qpos[:, 7] > close_thresh
    for t in range(len(closed) - 10):
        if closed[t:t+10].all():
            return t
    raise RuntimeError("Grasp not detected. Check close_thresh.")

def augment_qpos(qpos, grasp_t, delta_pos, delta_rot, grip_delta=0.0,
                 start_t=0, recovery_t=None, recovery_buffer=100):
    """
    Return (aug_qpos, recovery_t).

    Perturbation is fully applied at start_t, then linearly decays to identity
    by recovery_t (defaults to grasp_t - recovery_buffer). Stays at identity
    after recovery_t, leaving the grasp/place phases unchanged.

    grip_delta: signed offset added to the gripper openness during the
                perturbed phase (decays with alpha, clipped to [0, 1]).
    """
    T = len(qpos)
    if recovery_t is None:
        recovery_t = max(start_t, grasp_t - recovery_buffer)

    alphas = np.zeros(T)
    if recovery_t > start_t:
        alphas[start_t:recovery_t] = np.linspace(1.0, 0.0, recovery_t - start_t)

    slerp_fn = Slerp([0.0, 1.0],
                     Rotation.concatenate([Rotation.identity(), delta_rot]))

    z_floor = qpos[:, 2].min()  # never push the EE below its lowest demo point

    aug = qpos.copy()
    for t in range(T):
        a = alphas[t]
        if a == 0.0:
            continue
        pos_new, quat_new = apply_se3(
            slerp_fn(a), a * delta_pos,
            qpos[t, 0:3], qpos[t, 3:7])
        pos_new[2] = max(pos_new[2], z_floor)
        aug[t, 0:3] = pos_new
        aug[t, 3:7] = quat_new
        aug[t, 7]   = np.clip(qpos[t, 7] + a * grip_delta, 0.0, 1.0)
    return aug, recovery_t


# ── Sim replay ────────────────────────────────────────────────────────────────

def replay_in_sim(aug_qpos, camera_names, env_init_state, recovery_t,
                  start_t=0, task_name='umi_pick'):
    """Replay aug_qpos[start_t:] through the oracle sim, returning observations.

    The demo starts at the perturbed EE state at start_t — the policy sees only
    the recovery and task execution, not the perturbation being applied.
    Objects are frozen at their original positions until recovery_t so accidental
    contact during the recovery phase can't displace them before the grasp.
    """
    env = make_ee_sim_env(task_name, camera_names=camera_names)
    env.reset()
    np.copyto(env.physics.data.qpos, env_init_state)
    env.physics.forward()

    can_addr       = env.task._can_qpos_addr
    bowl_addr      = env.task._bowl_qpos_addr
    can_qpos_orig  = env_init_state[can_addr:can_addr + 7].copy()
    bowl_qpos_orig = env_init_state[bowl_addr:bowl_addr + 7].copy()

    # Build set of can/bowl geom ids for contact detection
    def _geom_ids(bid):
        start = env.physics.model.body_geomadr[bid]
        n     = env.physics.model.body_geomnum[bid]
        return set(range(start, start + n))

    can_geom_ids  = _geom_ids(env.task._can_bid)
    bowl_geom_ids = _geom_ids(env.task._bowl_bid)
    table_bid     = env.physics.model.name2id('table', 'body')
    table_geom_ids = _geom_ids(table_bid)
    object_geom_ids = can_geom_ids | bowl_geom_ids | table_geom_ids

    def _object_contact(physics):
        """True if any active contact involves the can, bowl, or table."""
        for k in range(physics.data.ncon):
            c = physics.data.contact[k]
            if c.geom1 in object_geom_ids or c.geom2 in object_geom_ids:
                return True
        return False

    sub_traj = aug_qpos[start_t:]   # only replay from start_t
    T_sub    = len(sub_traj)

    observations = []
    last_object_contact_i = -1   # last local index where unwanted contact occurred

    freeze_until = recovery_t + 20  # extra buffer so gripper clears objects before physics releases

    for i in range(T_sub):
        t_global = start_t + i
        if t_global <= freeze_until:
            # Freeze objects during and just after recovery
            np.copyto(env.physics.data.qpos[can_addr:can_addr + 7],  can_qpos_orig)
            np.copyto(env.physics.data.qpos[bowl_addr:bowl_addr + 7], bowl_qpos_orig)
            env.physics.data.qvel[:] = 0
            env.physics.forward()

        ts = env.step(sub_traj[i])
        observations.append(ts.observation)

        # Track latest unintended contact during recovery phase (up to freeze_until)
        if t_global <= freeze_until and _object_contact(env.physics):
            last_object_contact_i = i

        if i % 500 == 0:
            print(f"  replay {i}/{T_sub}")

    # Crop to start just after the last unwanted contact so the saved demo
    # never contains frames where the gripper touches objects unintentionally
    crop = last_object_contact_i + 1
    if crop > 0:
        print(f"  cropped {crop} steps due to unintended object contact")
    observations = observations[crop:]
    return observations, aug_qpos[start_t + crop:]


# ── Save ──────────────────────────────────────────────────────────────────────

def save_augmented(path, aug_qpos, observations, camera_names):
    tmp = path + '.tmp'
    with h5py.File(tmp, 'w', rdcc_nbytes=1024**2 * 2) as root:
        root.attrs['sim']       = True
        root.attrs['success']   = True
        root.attrs['augmented'] = True
        obs_grp = root.create_group('observations')
        obs_grp.create_dataset('qpos', data=aug_qpos)
        root.create_dataset('action', data=aug_qpos)
        img_grp = obs_grp.create_group('images')
        for cam in camera_names:
            arr = np.stack([o['images'][cam] for o in observations])
            img_grp.create_dataset(cam, data=arr, dtype='uint8',
                                   chunks=(1,) + arr.shape[1:])
    os.replace(tmp, path)
    print(f"Saved → {path}")


# ── Visualisation ─────────────────────────────────────────────────────────────

def visualise_trajectory(qpos, aug_qpos, grasp_t, out_dir):
    T    = len(qpos)
    time = np.arange(T) * DT

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for i, (ax, lbl) in enumerate(zip(axes, ['x', 'y', 'z'])):
        ax.plot(time, qpos[:, i],     lw=1.5, label=f'original {lbl}')
        ax.plot(time, aug_qpos[:, i], lw=1.5, ls='--', label=f'augmented {lbl}')
        ax.axvline(grasp_t * DT, color='k', ls=':', lw=1, label='grasp')
        ax.set_ylabel(f'{lbl} (m)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel('time (s)')
    fig.suptitle('Original vs Augmented EE trajectory')
    out = os.path.join(out_dir, 'trajectory_comparison.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved trajectory plot → {out}")

def visualise_frames(qpos, aug_qpos, grasp_t, observations_aug, out_dir):
    T = len(qpos)
    sample_ts = [0, grasp_t // 2, grasp_t, grasp_t + (T - grasp_t) // 2, T - 1]
    sample_ts = sorted(set(min(t, T - 1) for t in sample_ts))

    fig, axes = plt.subplots(1, len(sample_ts), figsize=(4 * len(sample_ts), 4))
    for ax, t in zip(axes, sample_ts):
        ax.imshow(observations_aug[t]['images']['ee'])
        ax.set_title(f't={t} ({t*DT:.1f}s)')
        ax.axis('off')
    fig.suptitle('Augmented trajectory frames')
    out = os.path.join(out_dir, 'augmented_frames.png')
    fig.savefig(out, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved frame comparison → {out}")


# ── Video ─────────────────────────────────────────────────────────────────────

def _put_text_bg(img, text, pos, font_scale=0.7, color=(255,255,255),
                 thickness=2, bg_color=(0,0,0)):
    """Draw text with a solid background box for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    cv2.rectangle(img, (x - 4, y - th - 4), (x + tw + 4, y + baseline + 4),
                  bg_color, -1)
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness,
                cv2.LINE_AA)

def make_side_by_side_video(orig_images, aug_observations, grasp_t, out_path, fps=50,
                            recovery_buffer=300):
    """Write a side-by-side mp4: original (left) vs augmented (right)."""
    T          = min(len(orig_images), len(aug_observations))
    recovery_t = max(0, grasp_t - recovery_buffer)
    H, W       = orig_images[0].shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (W * 2, H))

    for t in range(T):
        orig_bgr = cv2.cvtColor(orig_images[t], cv2.COLOR_RGB2BGR)
        aug_bgr  = cv2.cvtColor(aug_observations[t]['images']['ee'], cv2.COLOR_RGB2BGR)

        # ── Left panel: original ──────────────────────────────────────────────
        _put_text_bg(orig_bgr, 'ORIGINAL DEMONSTRATION', (10, 30),
                     font_scale=0.8, color=(255, 255, 255))
        _put_text_bg(orig_bgr, f't={t}  {t*DT:.2f}s', (10, H - 12),
                     font_scale=0.55, color=(200, 200, 200))

        # ── Right panel: augmented ────────────────────────────────────────────
        if t < recovery_t:
            phase_label = 'PERTURBED APPROACH'
            phase_color = (80, 80, 255)
            border_color = (0, 0, 220)
            status = f'SE(3) offset decaying to zero  ({(1 - t/recovery_t)*100:.0f}% remaining)'
        elif t < grasp_t:
            phase_label = 'RECOVERED  (perturbation = 0)'
            phase_color = (80, 220, 80)
            border_color = (0, 180, 0)
            status = f'Back on original trajectory — {grasp_t - t} steps before grasp'
        else:
            phase_label = 'ORIGINAL TRAJECTORY (spliced)'
            phase_color = (200, 200, 200)
            border_color = None
            status = 'Grasp + place unchanged from demo'

        _put_text_bg(aug_bgr, f'AUGMENTED  |  {phase_label}', (10, 30),
                     font_scale=0.8, color=phase_color)
        _put_text_bg(aug_bgr, status, (10, H - 12),
                     font_scale=0.55, color=(200, 200, 200))

        if border_color is not None:
            cv2.rectangle(aug_bgr, (0, 0), (W-1, H-1), border_color, 5)

        # Grasp marker flash
        if t == grasp_t:
            cv2.rectangle(aug_bgr, (0, 0), (W-1, H-1), (0, 220, 0), 5)

        frame = np.concatenate([orig_bgr, aug_bgr], axis=1)
        writer.write(frame)

        if t % 500 == 0:
            print(f"  video {t}/{T}")

    writer.release()
    print(f"Saved video → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default=
        '/local/real/jgoler/UMI-on-Air/data/bc/umi_pick/demonstration/episode_0.hdf5')
    parser.add_argument('--output', default=
        '/local/real/jgoler/UMI-on-Air/data/bc/umi_pick/augmented/episode_0_aug.hdf5')
    parser.add_argument('--delta_pos', nargs=3, type=float,
                        default=[0.05, 0.03, 0.02],
                        metavar=('X', 'Y', 'Z'),
                        help='Translation perturbation at t=0 in metres')
    parser.add_argument('--delta_rot_deg', nargs=3, type=float,
                        default=[5.0, 3.0, 2.0],
                        metavar=('R', 'P', 'Y'),
                        help='Rotation perturbation at t=0 in degrees (roll pitch yaw)')
    parser.add_argument('--no_replay', action='store_true',
                        help='Skip sim replay; only plot trajectory math')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out_dir = os.path.dirname(args.output)

    # Load
    print(f"Loading {args.input} ...")
    with h5py.File(args.input, 'r') as f:
        qpos           = f['observations/qpos'][:]
        orig_images    = f['observations/images/ee'][:]
        env_init_state = f['env_init_state'][:]
        camera_names   = list(f['observations/images'].keys())
    print(f"  T={len(qpos)}, cameras={camera_names}")

    # Grasp time
    grasp_t = detect_grasp_time(qpos)
    print(f"  Grasp at t={grasp_t} ({grasp_t*DT:.2f}s)")

    # Perturbation
    delta_pos = np.array(args.delta_pos)
    delta_rot = Rotation.from_euler('xyz', args.delta_rot_deg, degrees=True)
    print(f"  delta_pos={delta_pos}  delta_rot_deg={args.delta_rot_deg}")

    aug_qpos, recovery_t = augment_qpos(qpos, grasp_t, delta_pos, delta_rot)
    print(f"  Perturbation applied. recovery_t={recovery_t} ({recovery_t*DT:.2f}s)")

    # Trajectory plot (no sim needed)
    visualise_trajectory(qpos, aug_qpos, grasp_t, out_dir)

    if args.no_replay:
        print("Skipping sim replay (--no_replay).")
        return

    # Sim replay
    print("Replaying in sim ...")
    observations, aug_qpos_save = replay_in_sim(
        aug_qpos, camera_names, env_init_state, recovery_t)

    # Save (aug_qpos_save is already sliced from start_t)
    save_augmented(args.output, aug_qpos_save, observations, camera_names)

    # Frame visualisation
    visualise_frames(qpos, aug_qpos, grasp_t, observations, out_dir)

    # Video
    video_path = os.path.join(out_dir, 'side_by_side.mp4')
    make_side_by_side_video(orig_images, observations, grasp_t, video_path)


if __name__ == '__main__':
    main()

"""
augment_batch.py  -  Generate many augmented trajectories from one demo.

Produces:
  - augmented/batch/episode_aug_<i>.hdf5  for each variant
  - augmented/batch/grid_video.mp4        tiled original + all variants

Usage
-----
    MUJOCO_GL=egl python augment_batch.py \
        --input /local/real/jgoler/UMI-on-Air/data/bc/umi_pick/demonstration/episode_2.hdf5
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
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.spatial.transform import Rotation, Slerp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from constants import DT
from ee_sim_env import make_ee_sim_env
from augment_episode import (
    detect_grasp_time, augment_qpos, replay_in_sim, save_augmented
)

MIN_BUFFER   = 50   # min steps between recovery_t and grasp_t
MIN_DURATION = 40   # min steps a perturbation must last

def build_perturbations(grasp_t, n=15, seed=0):
    """
    Sample n diverse perturbations. Each starts at a perturbed state and
    recovers back to the original at a different time, so perturbations
    span the full pre-grasp window.

    Per-variant randomisation:
      - start_t      : when the perturbation begins  [0, 0.65 * grasp_t]
      - recovery_t   : when it fully recovers        [start_t + MIN_DURATION,
                                                       grasp_t - MIN_BUFFER]
      - translation  : random full-3D direction, magnitude 5–22 cm
      - rotation     : independent random axis, magnitude 5–25 deg
      - grip_delta   : ±0 to 0.3 offset on normalised gripper openness
    """
    rng = np.random.RandomState(seed)
    variants = []
    for i in range(n):
        # Timing
        start_t    = int(rng.uniform(0, grasp_t * 0.65))
        min_rec    = start_t + MIN_DURATION
        max_rec    = grasp_t - MIN_BUFFER
        recovery_t = int(rng.uniform(min_rec, max(min_rec + 1, max_rec)))

        # Translation: full 3D random direction
        vec     = rng.randn(3)
        vec[2] *= 0.4           # reduce Z component so we don't go through floor
        pos_dir  = vec / np.linalg.norm(vec)
        pos_m    = rng.uniform(0.10, 0.40)

        # Rotation: independent random axis, large magnitude
        rot_axis = rng.randn(3)
        rot_axis /= np.linalg.norm(rot_axis)
        rot_deg  = rng.uniform(15.0, 45.0)

        # Gripper: random partial open/close offset
        grip_delta = rng.uniform(-0.4, 0.4)

        delta_pos  = pos_dir * pos_m
        delta_rot  = Rotation.from_rotvec(rot_axis * np.deg2rad(rot_deg))
        label      = f'aug{i:02d}'
        variants.append((label, delta_pos, delta_rot, grip_delta, start_t, recovery_t))
    return variants


# ── Grid video ────────────────────────────────────────────────────────────────

GRID_COLS  = 4
GRID_ROWS  = 4           # 16 cells: 1 original + 15 augmented
CELL_W     = 320
CELL_H     = 240

def _resize(img_rgb, w=CELL_W, h=CELL_H):
    return cv2.resize(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), (w, h))

def _put(cell, text, x, y, scale=0.42, color=(255,255,255), thickness=1):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), bl = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(cell, (x - 2, y - th - 2), (x + tw + 2, y + bl + 2), (0,0,0), -1)
    cv2.putText(cell, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)

def _annotate_original(cell, t):
    _put(cell, 'ORIGINAL DEMONSTRATION', 4, 16, scale=0.45, color=(220,220,220))
    _put(cell, f't={t}  {t*DT:.1f}s', 4, CELL_H - 6, scale=0.38, color=(180,180,180))

def _annotate_augmented(cell, t, label, delta_pos, delta_rot_deg,
                        grip_delta, start_t, recovery_t, grasp_t):
    pos_mag = np.linalg.norm(delta_pos)
    _put(cell, label, 4, 16, scale=0.42, color=(200,200,200))
    _put(cell, f'{pos_mag*100:.0f}cm  {delta_rot_deg:.0f}deg  grip{grip_delta:+.2f}',
         4, 33, scale=0.33, color=(160,160,160))
    _put(cell, f'start={start_t*DT:.1f}s  rec={recovery_t*DT:.1f}s',
         4, 48, scale=0.33, color=(140,140,140))

    if t < start_t:
        phase  = f'original  (perturb in {(start_t-t)*DT:.1f}s)'
        color  = (160, 160, 160)
        border = None
    elif t < recovery_t:
        pct    = (1 - (t - start_t) / max(1, recovery_t - start_t)) * 100
        phase  = f'PERTURBED  {pct:.0f}% left'
        color  = (60, 60, 230)
        border = (0, 0, 180)
    elif t < grasp_t:
        phase  = f'RECOVERED  ({(grasp_t-t)*DT:.1f}s to grasp)'
        color  = (60, 200, 60)
        border = (0, 160, 0)
    else:
        phase  = 'spliced back'
        color  = (160, 160, 160)
        border = None

    _put(cell, phase, 4, CELL_H - 6, scale=0.35, color=color)
    if border is not None:
        cv2.rectangle(cell, (0,0), (CELL_W-1, CELL_H-1), border, 3)

def make_grid_video(orig_images, all_observations, labels, perturbation_info,
                    grasp_t, out_path, fps=50):
    """
    orig_images       : (T, H, W, 3)
    all_observations  : list of N obs lists
    labels            : list of N strings
    perturbation_info : list of N (delta_pos, delta_rot_deg, start_t, recovery_t) tuples
    """
    # Each augmented demo starts at start_t so has length T-start_t.
    # Align to the longest (original) by padding shorter ones with black.
    T_orig = len(orig_images)
    T_max  = T_orig
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'),
                          fps, (GRID_COLS * CELL_W, GRID_ROWS * CELL_H))

    blank = np.zeros((CELL_H, CELL_W, 3), dtype=np.uint8)

    for t in range(T_max):
        cells = []

        # Cell 0: original (full length)
        cell = _resize(orig_images[t])
        _annotate_original(cell, t)
        cells.append(cell)

        # Cells 1..15: augmented variants — offset by start_t
        for obs_list, lbl, (delta_pos, delta_rot_deg, grip_delta, start_t, recovery_t) in zip(
                all_observations, labels, perturbation_info):
            aug_t = t - start_t   # local index within this augmented demo
            if aug_t < 0 or aug_t >= len(obs_list):
                cell = blank.copy()
                _put(cell, f'{lbl} (not started)' if aug_t < 0 else f'{lbl} (done)',
                     4, CELL_H // 2, scale=0.35, color=(100,100,100))
            else:
                cell = _resize(obs_list[aug_t]['images']['ee'])
                _annotate_augmented(cell, t, lbl, delta_pos, delta_rot_deg,
                                    grip_delta, start_t, recovery_t, grasp_t)
            cells.append(cell)

        # Arrange into grid
        rows = []
        for r in range(GRID_ROWS):
            row_cells = cells[r * GRID_COLS:(r + 1) * GRID_COLS]
            while len(row_cells) < GRID_COLS:
                row_cells.append(np.zeros((CELL_H, CELL_W, 3), dtype=np.uint8))
            rows.append(np.concatenate(row_cells, axis=1))
        frame = np.concatenate(rows, axis=0)
        out.write(frame)

        if t % 200 == 0:
            print(f"  grid video {t}/{T_max}")

    out.release()
    print(f"Saved grid video → {out_path}")


# ── 3-D trajectory plot ───────────────────────────────────────────────────────

def make_3d_plot(qpos, all_aug_qpos, variants, grasp_t, out_path):
    """
    3-D scatter/line plot showing the original trajectory and all augmented
    variants in XYZ EE space.

    For each augmented variant:
      - perturbed segment  [start_t : recovery_t]  → coloured line
      - recovery segment   [recovery_t : grasp_t]  → same colour, dashed
      - perturb start point → large dot
      - recovery point     → small marker where it rejoins original

    Original trajectory is shown in black, with the grasp point marked.
    """
    fig = plt.figure(figsize=(14, 10))
    ax  = fig.add_subplot(111, projection='3d')

    orig_xyz = qpos[:, 0:3]

    # Original trajectory
    ax.plot(orig_xyz[:, 0], orig_xyz[:, 1], orig_xyz[:, 2],
            color='black', lw=2, zorder=5, label='original')
    ax.scatter(*orig_xyz[0],      color='black', s=60, marker='s', zorder=6)
    ax.scatter(*orig_xyz[grasp_t], color='green', s=80, marker='*', zorder=6,
               label=f'grasp (t={grasp_t})')

    cmap = plt.get_cmap('tab20')
    n    = len(variants)

    for idx, ((label, _, _, _, start_t, recovery_t), aug_qpos) in enumerate(
            zip(variants, all_aug_qpos)):

        color = cmap(idx / max(n - 1, 1))
        xyz   = aug_qpos[:, 0:3]

        # Indices relative to full-length aug_qpos (same length as qpos)
        s, r, g = start_t, recovery_t, grasp_t

        # Perturbed phase: start_t → recovery_t
        if r > s:
            ax.plot(xyz[s:r, 0], xyz[s:r, 1], xyz[s:r, 2],
                    color=color, lw=1.5, alpha=0.9)

        # Recovery phase: recovery_t → grasp_t (dashed, lighter)
        if g > r:
            ax.plot(xyz[r:g, 0], xyz[r:g, 1], xyz[r:g, 2],
                    color=color, lw=1.0, ls='--', alpha=0.6)

        # Start dot (perturbed position)
        ax.scatter(*xyz[s], color=color, s=50, marker='o', zorder=7)

        # Recovery dot
        ax.scatter(*xyz[r], color=color, s=20, marker='^', zorder=7)

        # Label only the start dot
        ax.text(xyz[s, 0], xyz[s, 1], xyz[s, 2], f' {label}',
                fontsize=6, color=color, va='bottom')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('EE trajectories: original (black) + augmented variants\n'
                 'Solid = perturbed phase  |  Dashed = recovery phase  |  '
                 'Circle = perturb start  |  Triangle = recovery point')

    # Compact legend: only original + grasp marker
    ax.legend(loc='upper left', fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved 3-D trajectory plot → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default=
        '/local/real/jgoler/UMI-on-Air/data/bc/umi_pick/demonstration/episode_2.hdf5')
    parser.add_argument('--out_dir', default=
        '/local/real/jgoler/UMI-on-Air/data/bc/umi_pick/augmented/batch')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load demo
    print(f"Loading {args.input} ...")
    with h5py.File(args.input, 'r') as f:
        qpos           = f['observations/qpos'][:]
        orig_images    = f['observations/images/ee'][:]
        env_init_state = f['env_init_state'][:]
        camera_names   = list(f['observations/images'].keys())

    grasp_t = detect_grasp_time(qpos)
    print(f"  T={len(qpos)}, grasp_t={grasp_t} ({grasp_t*DT:.1f}s)")

    variants = build_perturbations(grasp_t)
    print(f"  Generating {len(variants)} augmented trajectories ...\n")

    all_observations  = []
    all_aug_qpos      = []   # full-length aug_qpos for 3-D plot
    labels            = []
    perturbation_info = []

    for i, (label, delta_pos, delta_rot, grip_delta, start_t, var_recovery_t) in enumerate(variants):
        rot_deg = np.degrees(delta_rot.magnitude())
        print(f"[{i+1:02d}/{len(variants)}] {label}  "
              f"pos={np.linalg.norm(delta_pos)*100:.1f}cm  rot={rot_deg:.1f}deg  "
              f"grip={grip_delta:+.2f}  start={start_t*DT:.1f}s  recovery={var_recovery_t*DT:.1f}s")

        aug_qpos, _ = augment_qpos(
            qpos, grasp_t, delta_pos, delta_rot,
            grip_delta=grip_delta, start_t=start_t, recovery_t=var_recovery_t)
        obs, aug_qpos_save = replay_in_sim(
            aug_qpos, camera_names, env_init_state, var_recovery_t,
            start_t=start_t)

        hdf5_path = os.path.join(args.out_dir, f'episode_aug_{i:02d}_{label}.hdf5')
        save_augmented(hdf5_path, aug_qpos_save, obs, camera_names)

        all_observations.append(obs)
        all_aug_qpos.append(aug_qpos)   # full-length, before cropping
        labels.append(label)
        perturbation_info.append((delta_pos, rot_deg, grip_delta, start_t, var_recovery_t))
        print()

    print("Building 3-D trajectory plot ...")
    plot_path = os.path.join(args.out_dir, 'trajectories_3d.png')
    make_3d_plot(qpos, all_aug_qpos, variants, grasp_t, plot_path)

    print("Building grid video ...")
    video_path = os.path.join(args.out_dir, 'grid_video.mp4')
    make_grid_video(orig_images, all_observations, labels, perturbation_info,
                    grasp_t, video_path)
    print("Done.")


if __name__ == '__main__':
    main()

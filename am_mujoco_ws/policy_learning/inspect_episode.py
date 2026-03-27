"""Quick sanity-check for a recorded umi_pick episode."""
import sys
import h5py
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else \
    '/local/real/jgoler/UMI-on-Air/data/bc/umi_pick/demonstration/episode_0.hdf5'

with h5py.File(path, 'r') as f:
    qpos   = f['observations/qpos'][:]   # (T, 8)
    action = f['action'][:]              # (T, 8)
    img    = f['observations/images/ee'] # (T, H, W, 3)

    print(f"=== {path} ===")
    print(f"T (timesteps)     : {qpos.shape[0]}")
    print(f"qpos   shape      : {qpos.shape}")
    print(f"action shape      : {action.shape}")
    print(f"image  shape      : {img.shape}")
    print()

    print("--- qpos columns: [x, y, z, qw, qx, qy, qz, grip_norm] ---")
    labels = ['x', 'y', 'z', 'qw', 'qx', 'qy', 'qz', 'grip']
    for i, label in enumerate(labels):
        col = qpos[:, i]
        print(f"  {label:6s}  min={col.min():.4f}  max={col.max():.4f}  "
              f"start={col[0]:.4f}  end={col[-1]:.4f}")

    print()
    print("--- action columns ---")
    for i, label in enumerate(labels):
        col = action[:, i]
        print(f"  {label:6s}  min={col.min():.4f}  max={col.max():.4f}")

    print()
    # Detect approximate grasp time: first sustained gripper close
    grip = qpos[:, 7]
    close_thresh = 0.6  # normalized: 0=open, ~0.9=closed
    closed = grip > close_thresh
    grasp_t = None
    for t in range(len(closed) - 10):
        if closed[t:t+10].all():
            grasp_t = t
            break
    if grasp_t is not None:
        print(f"Approximate grasp timestep: {grasp_t}  ({grasp_t * 0.02:.2f}s)")
    else:
        print("Grasp not detected (gripper never sustained close)")

    print()
    print(f"attrs: { {k: v for k, v in f.attrs.items()} }")

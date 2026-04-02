"""
depth_warp.py  -  RGB-D point cloud warping for view synthesis.

Given an RGB image + depth map and a camera intrinsics matrix K, this module:
  1. Unprojects each pixel to a 3-D coloured point cloud.
  2. Applies an SE(3) transform (delta pose) to simulate a new camera viewpoint.
  3. Reprojects back to a 2-D image with z-buffer occlusion handling.
  4. Inpaints holes left by disocclusions.

Camera convention
-----------------
  - Pinhole model, no distortion.
  - Z points forward (into the scene), X right, Y down.
  - depth values are metric (metres), perpendicular to image plane (Z depth,
    not ray depth).

Usage
-----
    python depth_warp.py   # runs a self-contained demo using a sim episode

API
---
    K          = camera_K_from_fov(fovy_deg, H, W)
    pointcloud = unproject(rgb, depth, K)           # (N,6) [x,y,z,r,g,b]
    warped     = reproject(pointcloud, K, H, W)     # (H,W,3) uint8
    warped     = warp_image(rgb, depth, K, T_delta) # convenience wrapper
"""

import os
import sys
import numpy as np
import cv2

# ── Camera helpers ─────────────────────────────────────────────────────────────

def camera_K_from_fov(fovy_deg, H, W):
    """Build a 3x3 pinhole intrinsics matrix from vertical FOV + image size."""
    fy = H / (2.0 * np.tan(np.deg2rad(fovy_deg) / 2.0))
    fx = fy  # square pixels
    cx = W / 2.0
    cy = H / 2.0
    return np.array([[fx,  0, cx],
                     [ 0, fy, cy],
                     [ 0,  0,  1]], dtype=np.float64)


# ── Point cloud ────────────────────────────────────────────────────────────────

def unproject(rgb, depth, K):
    """
    Unproject an RGB-D image to a coloured 3-D point cloud.

    Parameters
    ----------
    rgb   : (H, W, 3) uint8
    depth : (H, W)    float32  metric depth in metres
    K     : (3, 3)    camera intrinsics

    Returns
    -------
    pc : (N, 6) float32  columns [x, y, z, r, g, b]
         Only pixels with depth > 0 are included.
    """
    H, W = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    v_idx, u_idx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')  # (H,W)

    z = depth.astype(np.float32)
    x = (u_idx - cx) * z / fx
    y = (v_idx - cy) * z / fy

    mask = z > 0
    xyz  = np.stack([x[mask], y[mask], z[mask]], axis=1)   # (N, 3)
    rgb_f = rgb[mask].astype(np.float32)                    # (N, 3)

    return np.concatenate([xyz, rgb_f], axis=1)             # (N, 6)


def transform_pointcloud(pc, T):
    """
    Apply a 4x4 SE(3) transform to a point cloud.

    Parameters
    ----------
    pc : (N, 6) [x, y, z, r, g, b]
    T  : (4, 4) SE(3) matrix  (transform from old camera frame to new)

    Returns
    -------
    pc_new : (N, 6) transformed point cloud (colours unchanged)
    """
    xyz  = pc[:, :3]                        # (N, 3)
    ones = np.ones((len(xyz), 1), dtype=np.float32)
    xyz_h = np.concatenate([xyz, ones], axis=1)  # (N, 4)
    xyz_new = (T @ xyz_h.T).T[:, :3]            # (N, 3)
    return np.concatenate([xyz_new, pc[:, 3:]], axis=1)


def reproject(pc, K, H, W):
    """
    Reproject a coloured point cloud onto a 2-D image plane using z-buffering.

    Parameters
    ----------
    pc : (N, 6) [x, y, z, r, g, b]
    K  : (3, 3) camera intrinsics
    H, W : output image size

    Returns
    -------
    rgb_out  : (H, W, 3) uint8   — black where no point projects
    hole_mask: (H, W)    bool    — True where no point landed (needs inpainting)
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    x, y, z = pc[:, 0], pc[:, 1], pc[:, 2]

    # Keep only finite points in front of camera
    valid = (z > 1e-4) & np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[valid], y[valid], z[valid]
    colors = pc[valid, 3:]

    # Project
    u = (fx * x / z + cx).astype(np.float32)
    v = (fy * y / z + cy).astype(np.float32)

    # Round to pixel
    u_i = np.round(u).astype(np.int32)
    v_i = np.round(v).astype(np.int32)

    in_bounds = (u_i >= 0) & (u_i < W) & (v_i >= 0) & (v_i < H)
    u_i, v_i, z, colors = u_i[in_bounds], v_i[in_bounds], z[in_bounds], colors[in_bounds]

    # Z-buffer: for each pixel keep the closest point
    z_buf  = np.full((H, W), np.inf, dtype=np.float32)
    rgb_out = np.zeros((H, W, 3), dtype=np.float32)

    # Sort by depth descending so closer points overwrite farther ones
    order = np.argsort(z)[::-1]
    u_i, v_i, z, colors = u_i[order], v_i[order], z[order], colors[order]

    rgb_out[v_i, u_i] = colors
    z_buf[v_i, u_i]   = z

    hole_mask = np.isinf(z_buf)
    rgb_out   = rgb_out.clip(0, 255).astype(np.uint8)

    return rgb_out, hole_mask


def inpaint_holes(rgb, hole_mask):
    """
    Fill disocclusion holes using OpenCV's Navier-Stokes inpainting.

    Parameters
    ----------
    rgb       : (H, W, 3) uint8
    hole_mask : (H, W)    bool   True = hole

    Returns
    -------
    (H, W, 3) uint8 with holes filled
    """
    mask_u8 = hole_mask.astype(np.uint8) * 255
    return cv2.inpaint(rgb, mask_u8, inpaintRadius=3, flags=cv2.INPAINT_NS)


# ── Gripper segmentation ──────────────────────────────────────────────────────

GRIPPER_GEOM_NAMES = ('gripper_base_geom', 'lfinger_geom', 'rfinger_geom')

def get_gripper_mask(physics, H, W, camera_id='ee'):
    """
    Return a boolean mask (H, W) that is True wherever a gripper geom is the
    closest object to the camera, using MuJoCo segmentation rendering.

    segmentation render: channel 0 = geom_id of frontmost geom (-1 = sky/background)
    """
    seg = physics.render(H, W, camera_id=camera_id, segmentation=True)
    geom_ids = {physics.model.name2id(name, 'geom') for name in GRIPPER_GEOM_NAMES}
    # seg[:, :, 0] is the frontmost geom id at each pixel
    return np.isin(seg[:, :, 0], list(geom_ids))


# ── Convenience wrapper ────────────────────────────────────────────────────────

def warp_image(rgb, depth, K, T_delta, inpaint=False):
    """
    Synthesise what the camera would see after applying SE(3) transform T_delta.

    Parameters
    ----------
    rgb     : (H, W, 3) uint8
    depth   : (H, W)    float32  metres
    K       : (3, 3)    camera intrinsics
    T_delta : (4, 4)    SE(3) delta transform (new_cam_from_old_cam)
    inpaint : bool      if True, fill disocclusion holes; if False, leave black

    Returns
    -------
    warped : (H, W, 3) uint8
    """
    H, W  = rgb.shape[:2]
    pc    = unproject(rgb, depth, K)
    pc    = transform_pointcloud(pc, T_delta)
    warped, holes = reproject(pc, K, H, W)
    if inpaint:
        warped = inpaint_holes(warped, holes)
    return warped


# ── Multi-frame world point cloud ─────────────────────────────────────────────

def _get_cam_to_world(physics, cam_name='ee'):
    """
    4x4 transform from pinhole camera frame (Z-in, Y-down) to world frame.

    MuJoCo stores cam_xmat as R such that v_world = R @ v_mujoco_cam,
    where MuJoCo camera convention is X-right, Y-up, Z-out.
    We flip Y and Z to match the pinhole convention used by unproject/reproject.
    """
    cam_id = physics.model.name2id(cam_name, 'camera')
    pos    = physics.data.cam_xpos[cam_id].copy()
    R_mj   = physics.data.cam_xmat[cam_id].reshape(3, 3).copy()
    # MuJoCo cam: X right, Y up, Z out → pinhole: X right, Y down, Z in
    flip   = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)
    T      = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_mj @ flip
    T[:3,  3] = pos
    return T


def _set_sim_frame(env, env_init_state, qpos_frame,
                   gripper_open, gripper_close, settle_steps=50):
    """
    Set sim to env_init_state (objects) + EE/gripper from qpos_frame.
    Runs settle_steps physics steps so servo actuators reach their target.
    Mocap pose is pinned each step to prevent drift.
    """
    np.copyto(env.physics.data.qpos, env_init_state)
    grip = float(np.clip(qpos_frame[7], 0.0, 1.0))
    ctrl = gripper_close + (1.0 - grip) * (gripper_open - gripper_close)
    env.physics.data.ctrl[env.task.lfinger_actuator_id] = ctrl
    env.physics.data.ctrl[env.task.rfinger_actuator_id] = ctrl
    for _ in range(settle_steps):
        env.physics.data.mocap_pos[0]  = qpos_frame[0:3]
        env.physics.data.mocap_quat[0] = qpos_frame[3:7]
        env.physics.step()
    env.physics.forward()


def build_world_pointcloud(episode_path, frame_step=5, H=480, W=640, fovy=90.0,
                            task_name='umi_pick', cam_name='ee'):
    """
    Build a merged world-frame colored point cloud from a demo episode.

    Subsamples every frame_step frames. At each frame, renders RGB + depth from
    sim, excludes gripper pixels, unprojects to camera frame, and transforms to
    world frame. All frames are concatenated into one cloud.

    Object positions are reset to env_init_state at every frame (no object motion
    replay), which is accurate for pre-grasp frames where objects are static.

    Parameters
    ----------
    episode_path : str    path to HDF5 demo file
    frame_step   : int    use every Nth frame
    H, W         : int    render resolution
    fovy         : float  vertical FOV in degrees

    Returns
    -------
    pc_world : (N, 6) float32  [x, y, z, r, g, b] in world frame
    """
    import h5py
    from constants import UMI_GRIPPER_OPEN, UMI_GRIPPER_CLOSE
    from ee_sim_env import make_ee_sim_env

    with h5py.File(episode_path, 'r') as f:
        qpos_all       = f['observations/qpos'][:]
        env_init_state = f['env_init_state'][:]

    K        = camera_K_from_fov(fovy, H, W)
    T_demo   = len(qpos_all)
    t_frames = list(range(0, T_demo, frame_step))

    env = make_ee_sim_env(task_name, camera_names=[cam_name])
    env.reset()

    all_points = []
    for i, t in enumerate(t_frames):
        print(f"  point cloud: frame {i+1}/{len(t_frames)}  (t={t})", flush=True)
        _set_sim_frame(env, env_init_state, qpos_all[t],
                       UMI_GRIPPER_OPEN, UMI_GRIPPER_CLOSE)

        rgb          = env.physics.render(H, W, camera_id=cam_name)
        depth        = env.physics.render(H, W, camera_id=cam_name, depth=True)
        gripper_mask = get_gripper_mask(env.physics, H, W, camera_id=cam_name)
        depth[gripper_mask] = 0.0

        pc_cam = unproject(rgb, depth, K)
        if len(pc_cam) == 0:
            continue

        T_c2w    = _get_cam_to_world(env.physics, cam_name)
        pc_world = transform_pointcloud(pc_cam, T_c2w)
        all_points.append(pc_world)

    return np.concatenate(all_points, axis=0).astype(np.float32)


def downsample_pointcloud(pc, max_points):
    """Randomly downsample a point cloud to at most max_points points."""
    if len(pc) <= max_points:
        return pc
    idx = np.random.choice(len(pc), max_points, replace=False)
    return pc[idx]


def render_augmented_trajectory(world_pc, aug_qpos, episode_path, K,
                                 H=480, W=640, task_name='umi_pick', cam_name='ee',
                                 add_gripper=True):
    """
    Render each frame of aug_qpos by reprojecting world_pc into the augmented
    camera pose. Only sets mocap + physics.forward() — no gripper settling —
    so this is fast.

    Parameters
    ----------
    world_pc     : (N, 6)  world-frame point cloud from build_world_pointcloud
    aug_qpos     : (T, 8)  augmented trajectory [x,y,z,qw,qx,qy,qz,grip]
    episode_path : str     HDF5 path (for env_init_state object positions)
    K            : (3, 3)  camera intrinsics

    Returns
    -------
    frames : list of (H, W, 3) uint8
    """
    import h5py
    from ee_sim_env import make_ee_sim_env

    with h5py.File(episode_path, 'r') as f:
        env_init_state = f['env_init_state'][:]

    env = make_ee_sim_env(task_name, camera_names=[cam_name])
    env.reset()
    np.copyto(env.physics.data.qpos, env_init_state)
    env.physics.forward()

    T      = len(aug_qpos)
    frames = []
    prev_cam_pos = None
    for t, qpos_t in enumerate(aug_qpos):
        env.step(qpos_t)   # drives mocap + steps physics → cam_xpos updates reliably

        T_c2w  = _get_cam_to_world(env.physics, cam_name)
        T_w2c  = np.linalg.inv(T_c2w)
        pc_cam = transform_pointcloud(world_pc, T_w2c)
        rgb, _ = reproject(pc_cam, K, H, W)

        if add_gripper:
            # Render gripper pixels from sim (already at augmented pose) and composite
            rgb_sim      = env.physics.render(H, W, camera_id=cam_name)
            gripper_mask = get_gripper_mask(env.physics, H, W, camera_id=cam_name)
            rgb[gripper_mask] = rgb_sim[gripper_mask]

        frames.append(rgb)

        if t == 0:
            prev_cam_pos = T_c2w[:3, 3].copy()
        if t == 1:
            delta = np.linalg.norm(T_c2w[:3, 3] - prev_cam_pos)
            print(f"  cam moved {delta*100:.2f}cm between frame 0 and 1", flush=True)

        print(f"\r  rendering {t+1}/{T}", end='', flush=True)
    print()
    print(f"  total aug frames: {len(frames)}", flush=True)

    return frames


# ── Demo ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    import h5py
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.spatial.transform import Rotation

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ee_sim_env import make_ee_sim_env
    from constants import UMI_GRIPPER_OPEN, UMI_GRIPPER_CLOSE

    parser = argparse.ArgumentParser()
    parser.add_argument('--episode', default=
        '/local/real/jgoler/UMI-on-Air/data/bc/umi_pick/demonstration/episode_2.hdf5')
    parser.add_argument('--out_dir', default=
        '/local/real/jgoler/UMI-on-Air/data/bc/umi_pick/augmented/batch')
    parser.add_argument('--mode', choices=['single', 'trajectory'], default='single',
        help='single: one-frame warp demo; trajectory: full augmented trajectory')
    parser.add_argument('--frame_step', type=int, default=5,
        help='[trajectory mode] subsample every Nth frame for world point cloud')
    args = parser.parse_args()

    EPISODE = args.episode
    FOVY    = 90.0
    H, W    = 480, 640
    K       = camera_K_from_fov(FOVY, H, W)
    os.makedirs(args.out_dir, exist_ok=True)

    # ── Single-frame warp demo ────────────────────────────────────────────────
    if args.mode == 'single':
        print(f"Camera K:\n{K}")
        FRAME = 100
        with h5py.File(EPISODE, 'r') as f:
            env_init_state = f['env_init_state'][:]
            qpos_all       = f['observations/qpos'][:]
            rgb_orig       = f['observations/images/ee'][FRAME]

        print("Rendering depth and gripper mask from sim ...")
        env = make_ee_sim_env('umi_pick', camera_names=['ee'])
        env.reset()
        _set_sim_frame(env, env_init_state, qpos_all[FRAME],
                       UMI_GRIPPER_OPEN, UMI_GRIPPER_CLOSE)

        rgb_sim      = env.physics.render(H, W, camera_id='ee')
        depth_orig   = env.physics.render(H, W, camera_id='ee', depth=True)
        gripper_mask = get_gripper_mask(env.physics, H, W, camera_id='ee')

        depth_no_grip          = depth_orig.copy()
        depth_no_grip[gripper_mask] = 0.0
        rgb_scene_only         = rgb_orig.copy()
        rgb_scene_only[gripper_mask] = 0

        delta_pos = np.array([0.30, 0.0, 0.0])
        delta_rot = Rotation.from_euler('xyz', [10, 15, 0], degrees=True)
        T_delta   = np.eye(4)
        T_delta[:3, :3] = delta_rot.as_matrix()
        T_delta[:3,  3] = delta_pos

        print("Warping ...")
        rgb_warped = warp_image(rgb_scene_only, depth_no_grip, K, T_delta)

        out_path = os.path.join(args.out_dir, 'depth_warp_demo.png')
        fig, axes = plt.subplots(1, 4, figsize=(24, 5))
        axes[0].imshow(rgb_orig);       axes[0].set_title('HDF5 original');             axes[0].axis('off')
        axes[1].imshow(rgb_sim);        axes[1].set_title('Sim render');                axes[1].axis('off')
        axes[2].imshow(rgb_scene_only); axes[2].set_title('Fingertip regions blacked'); axes[2].axis('off')
        axes[3].imshow(rgb_warped);     axes[3].set_title('Warped, gripper excluded');  axes[3].axis('off')
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved → {out_path}")

    # ── Full trajectory demo ──────────────────────────────────────────────────
    else:
        from augment_episode import detect_grasp_time, augment_qpos, replay_in_sim

        with h5py.File(EPISODE, 'r') as f:
            qpos_all       = f['observations/qpos'][:]
            env_init_state = f['env_init_state'][:]
            camera_names   = list(f['observations/images'].keys())

        # Build world point cloud from all frames of the demo
        print(f"Building world point cloud (frame_step={args.frame_step}) ...")
        world_pc = build_world_pointcloud(EPISODE, frame_step=args.frame_step,
                                          H=H, W=W, fovy=FOVY)
        print(f"World point cloud: {len(world_pc):,} points (before downsample)")
        world_pc = downsample_pointcloud(world_pc, max_points=500_000)
        print(f"World point cloud: {len(world_pc):,} points (after downsample)")

        # Generate an augmented trajectory
        grasp_t = detect_grasp_time(qpos_all)
        print(f"Grasp at t={grasp_t}")
        delta_pos = np.array([0.12, 0.08, 0.0])
        delta_rot = Rotation.from_euler('z', 20, degrees=True)
        aug_qpos, recovery_t = augment_qpos(qpos_all, grasp_t, delta_pos, delta_rot)
        print(f"Augmented trajectory: recovery_t={recovery_t}")

        # Replay augmented trajectory through sim to get ground-truth sim frames
        print("Replaying augmented trajectory in sim ...")
        sim_obs, _ = replay_in_sim(aug_qpos, camera_names, env_init_state, recovery_t)
        sim_frames = [o['images']['ee'] for o in sim_obs]
        print(f"  sim frames: {len(sim_frames)}")

        # Render augmented trajectory frames from world point cloud
        print("Rendering augmented trajectory from point cloud ...")
        pc_frames = render_augmented_trajectory(
            world_pc, aug_qpos, EPISODE, K, H=H, W=W)
        print(f"  point cloud frames: {len(pc_frames)}")

        # Align lengths (replay_in_sim may crop a few frames at the start)
        T_total = min(len(sim_frames), len(pc_frames))
        pc_offset = len(pc_frames) - T_total   # pc_frames starts from t=0 of aug_qpos
        sim_offset = len(sim_frames) - T_total  # sim_frames may be cropped
        print(f"  using {T_total} frames (offsets: sim={sim_offset}, pc={pc_offset})")

        # Side-by-side video: sim augmented (left) vs point cloud augmented (right)
        video_path = os.path.join(args.out_dir, 'depth_warp_trajectory.mp4')
        writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'),
                                 50, (W * 2, H))
        for i in range(T_total):
            sim_bgr = cv2.cvtColor(sim_frames[sim_offset + i], cv2.COLOR_RGB2BGR)
            pc_bgr  = cv2.cvtColor(pc_frames[pc_offset + i],  cv2.COLOR_RGB2BGR)
            cv2.putText(sim_bgr, 'Sim (oracle)',   (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.putText(pc_bgr,  'Point cloud',    (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            writer.write(np.concatenate([sim_bgr, pc_bgr], axis=1))
            print(f"\r  writing video {i+1}/{T_total}", end='', flush=True)
        writer.release()
        print(f"\nSaved video → {video_path}")

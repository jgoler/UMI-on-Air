"""Keyboard policy that uses OpenCV waitKey instead of pynput.
Works over SSH with X11 forwarding (no XRECORD extension needed).
"""
import numpy as np
from scipy.spatial.transform import Rotation as R

DEBUG = False


class CVKeyboardPolicy:
    """Keyboard teleoperation policy using OpenCV key capture."""

    def __init__(self, task_name=None):
        self.task_name = task_name or ""
        self.is_umi_task = "umi" in self.task_name.lower()

        # Initial state
        self.target_ee_state = np.array([0.0, 0.0, 1.2, 1.0, 0.0, 0.0, 0.0])
        self.gripper_status = 0.0  # 0=open, 1=closed

        # Movement settings
        self.speeds = [0.002, 0.005, 0.01]
        self.current_speed_idx = 1
        self.rotation_speed = 0.008

        # Key state – updated externally via process_key()
        self.current_keys = set()
        self.running = True
        self.recording = False

        # Hold movement keys for N frames after last press so a single tap
        # produces visible movement even over laggy X11 forwarding.
        self._hold_frames = 8  # ~160ms at 50Hz
        self._hold_counter = 0

        print("CV2 Keyboard Controls:")
        print("  WASD: Move horizontally")
        print("  Space: Move up  |  Shift (any): Move down")
        print("  Arrow keys: Rotate pitch/yaw")
        print("  Z/C: Roll left/right")
        print("  Q: Open gripper  |  E: Close gripper")
        print("  P: Start recording  |  R: Reset/restart")
        print("  1-3: Adjust speed  |  ESC: Exit")

    # ------------------------------------------------------------------
    # Key handling – called from the main loop with cv2.waitKeyEx result
    # ------------------------------------------------------------------
    _ARROW_UP = 65362
    _ARROW_DOWN = 65363
    _ARROW_LEFT = 65361
    _ARROW_RIGHT = 65363

    def process_key(self, key_code):
        """Process a cv2.waitKeyEx result. Returns a command string or None.

        Commands: 'exit', 'toggle_record', 'reset', 'delete_last', 'toggle_fullscreen'

        Movement keys persist for `_hold_frames` frames after the last press,
        so the user doesn't need to mash keys to move.
        """
        if key_code == -1:
            # No key pressed – decrement hold counter
            if self._hold_counter > 0:
                self._hold_counter -= 1
            else:
                self.current_keys.clear()
            return None

        # Mask platform differences
        key_code = key_code & 0xFFFF

        # One-shot commands
        if key_code == 27:  # ESC
            self.running = False
            return "exit"
        elif key_code == ord('p') or key_code == ord('P'):
            return "toggle_record"
        elif key_code == ord('r') or key_code == ord('R'):
            return "reset"
        elif key_code == ord('x') or key_code == ord('X'):
            return "delete_last"
        elif key_code == ord('f') or key_code == ord('F'):
            return "toggle_fullscreen"
        elif key_code in (ord('1'), ord('2'), ord('3')):
            self.current_speed_idx = key_code - ord('1')
            print(f"Speed set to level {self.current_speed_idx + 1} "
                  f"({self.speeds[self.current_speed_idx]:.3f} m/s)")
            return None

        # Movement keys – set key and reset hold counter
        self.current_keys.clear()
        self._hold_counter = self._hold_frames

        if key_code == ord('w') or key_code == ord('W'):
            self.current_keys.add('w')
        elif key_code == ord('s') or key_code == ord('S'):
            self.current_keys.add('s')
        elif key_code == ord('a') or key_code == ord('A'):
            self.current_keys.add('a')
        elif key_code == ord('d') or key_code == ord('D'):
            self.current_keys.add('d')
        elif key_code == ord(' '):
            self.current_keys.add('space')
        # Shift keys (left=65505, right=65506 on X11)
        elif key_code in (65505, 65506, 0xFFE1, 0xFFE2):
            self.current_keys.add('shift')
        elif key_code == ord('q') or key_code == ord('Q'):
            self.current_keys.add('q')
        elif key_code == ord('e') or key_code == ord('E'):
            self.current_keys.add('e')
        elif key_code == ord('z') or key_code == ord('Z'):
            self.current_keys.add('z')
        elif key_code == ord('c') or key_code == ord('C'):
            self.current_keys.add('c')
        # Arrow keys (X11 keysyms)
        elif key_code in (65362, 0xFF52):  # Up
            self.current_keys.add('up')
        elif key_code in (65364, 0xFF54):  # Down
            self.current_keys.add('down')
        elif key_code in (65361, 0xFF51):  # Left
            self.current_keys.add('left')
        elif key_code in (65363, 0xFF53):  # Right
            self.current_keys.add('right')
        else:
            if DEBUG:
                print(f"Unknown key code: {key_code}")

        return None

    # ------------------------------------------------------------------
    def is_recording(self):
        return self.recording

    def get_action(self):
        """Compute action from currently held keys."""
        if not self.running:
            import sys
            print("Exiting due to ESC key press")
            sys.exit(0)

        quat = self.target_ee_state[3:7]
        rot_matrix = R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
        forward = rot_matrix[:, 0]
        right = rot_matrix[:, 1]
        up = rot_matrix[:, 2]

        speed = self.speeds[self.current_speed_idx]

        # Movement
        if 'w' in self.current_keys:
            self.target_ee_state[0:3] += forward * speed
        if 's' in self.current_keys:
            self.target_ee_state[0:3] -= forward * speed
        if 'a' in self.current_keys:
            self.target_ee_state[0:3] += right * speed
        if 'd' in self.current_keys:
            self.target_ee_state[0:3] -= right * speed
        if 'space' in self.current_keys:
            self.target_ee_state[0:3] += up * speed * 0.5
        if 'shift' in self.current_keys:
            self.target_ee_state[0:3] -= up * speed * 0.5

        # Gripper
        grip_speed = 0.02
        if 'e' in self.current_keys:
            self.gripper_status = max(0.0, self.gripper_status - grip_speed)
        if 'q' in self.current_keys:
            self.gripper_status = min(1.0, self.gripper_status + grip_speed)

        # Rotation
        rotation_applied = False
        current_rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]])

        if 'up' in self.current_keys:
            current_rot = current_rot * R.from_rotvec([0, -self.rotation_speed, 0])
            rotation_applied = True
        if 'down' in self.current_keys:
            current_rot = current_rot * R.from_rotvec([0, self.rotation_speed, 0])
            rotation_applied = True
        if 'left' in self.current_keys:
            current_rot = current_rot * R.from_rotvec([0, 0, self.rotation_speed])
            rotation_applied = True
        if 'right' in self.current_keys:
            current_rot = current_rot * R.from_rotvec([0, 0, -self.rotation_speed])
            rotation_applied = True
        if 'z' in self.current_keys:
            current_rot = current_rot * R.from_rotvec([-self.rotation_speed, 0, 0])
            rotation_applied = True
        if 'c' in self.current_keys:
            current_rot = current_rot * R.from_rotvec([self.rotation_speed, 0, 0])
            rotation_applied = True

        if rotation_applied:
            new_quat = current_rot.as_quat()  # [x,y,z,w]
            self.target_ee_state[3:7] = [new_quat[3], new_quat[0], new_quat[1], new_quat[2]]
            qn = np.linalg.norm(self.target_ee_state[3:7])
            if qn > 1e-10:
                self.target_ee_state[3:7] /= qn

        action = np.zeros(8)
        action[0:3] = self.target_ee_state[0:3]
        action[3:7] = self.target_ee_state[3:7]
        action[7] = self.gripper_status
        return action

    def generate_trajectory(self, ts_first):
        qpos = ts_first.observation['qpos']
        self.target_ee_state[0:3] = qpos[0:3]
        self.target_ee_state[3:7] = qpos[3:7]
        self.gripper_status = qpos[7]

    def cleanup(self):
        pass

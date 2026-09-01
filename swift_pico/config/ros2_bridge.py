#!/usr/bin/env python3
"""
MuJoCo ↔ ROS2 bridge for Swift Pico.

Optimized Pipeline:
  mujoco_bridge (physics + viewer, main process)
      --[qpos/qvel/time via shared memory]--> render process (own EGL context + rclpy node)
                 →  /image_raw (800*800 rgb8, 30 Hz via RELIABLE QoS + cv_bridge)
                 →  /camera_info (fx=fy=866, cx=cy=500)
                 →  whycode node  →  /whycode_node/markers
                                 →  /whycode_node/image_out

The offscreen camera render + image publish runs in its own OS process (not a
thread) so it gets its own Python GIL and a genuinely separate CPU core from
the physics/viewer loop, keeping rendering off the same core as the rest of
the ROS2 bridge. It is forked before rclpy.init()/the interactive
viewer exist in this process, since forking after a live DDS participant or
GL context is created is unsafe.
"""

import sys, os, time, tty, termios, threading, select, math
import multiprocessing as mp
import numpy as np

import rclpy
from rclpy.node             import Node
from rclpy.qos              import QoSProfile, ReliabilityPolicy, HistoryPolicy
from actuator_msgs.msg      import Actuators
from nav_msgs.msg           import Odometry
from geometry_msgs.msg      import Quaternion
from rosgraph_msgs.msg      import Clock
from builtin_interfaces.msg import Time
from swift_msgs.msg         import SwiftMsgs
from sensor_msgs.msg        import Image as RosImage, CameraInfo

from cv_bridge              import CvBridge

import mujoco
import mujoco.viewer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from motor_dynamics import MotorDynamics, KT, OMEGA_MAX
from ament_index_python.packages import get_package_share_directory as _gpsd

# ── Model & simulation constants ─────────────────────────────────────────────
# Override with SWIFT_PICO_MODEL_XML (e.g. from a launch file's env) to load
# a different world, such as models/swift_pico/drone_arena.xml.
MODEL_PATH = os.environ.get(
    'SWIFT_PICO_MODEL_XML',
    os.path.join(_gpsd('swift_pico_description'), 'models', 'swift_pico', 'drone.xml'),
)
SIM_DT     = 0.005

ODOM_HZ   = 100
WHYCODE_HZ = 40

MOTOR_TIMEOUT = 0.5

# ── Camera constants — exact match to the real Swift Pico camera ────────────
CAM_W, CAM_H = 800, 800          
CAM_FOV_RAD  = 1.047               
CAM_FOV_DEG  = math.degrees(CAM_FOV_RAD)   

_FX = (CAM_W / 2.0) / math.tan(CAM_FOV_RAD / 2.0)   
_FY = _FX
_CX, _CY = CAM_W / 2.0, CAM_H / 2.0                 

# Viewer: sync every N physics steps so viewer.sync() never starves the loop
# Set to 10 to give the offscreen renderer and ROS threads priority
VIEWER_SYNC_EVERY = 10   


# ── ROS2 Node ─────────────────────────────────────────────────────────────────
class MuJoCoROS2Bridge(Node):

    def __init__(self):
        super().__init__('mujoco_swift_pico')

        best_effort = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=1)

        # Subscriptions
        self.create_subscription(Actuators, '/rotors/command/motor_speed',
                                 self._motor_cb, qos_profile=best_effort)
        self.create_subscription(SwiftMsgs, '/drone_command',
                                 self._drone_cmd_cb, 10)

        # Publishers
        self.odom_pub  = self.create_publisher(Odometry,   '/rotors/odometry', 10)
        self.clock_pub = self.create_publisher(Clock,       '/clock',           10)

        # 100 Hz odometry timer
        self.create_timer(1.0 / ODOM_HZ, self._pub_odometry)

        # Shared state
        self._state_lock = threading.Lock()
        self._state = dict(pos=np.zeros(3), quat_mj=np.array([1.,0.,0.,0.]),
                           vel=np.zeros(3), ang_vel=np.zeros(3), sim_time=0.0)

        self._cmd_lock      = threading.Lock()
        self._omega_cmd     = np.zeros(4)
        self._last_cmd_time = 0.0

        self._arm_lock = threading.Lock()
        self._armed    = False

        self.get_logger().info('MuJoCo ROS2 bridge ready')
        self.get_logger().info(
            f'  Camera: {CAM_W}×{CAM_H} px, hfov={CAM_FOV_RAD:.3f} rad, '
            f'fx=fy={_FX:.2f} px  (rendered in a separate process)')
        self.get_logger().info(
            '  /image_raw + /camera_info  →  whycode node  →  /whycode_node/markers')

    # ── Motor / arm callbacks ─────────────────────────────────────────────────
    #Update the commanded motor speeds (rad/s) from the /rotors/command/motor_speed topic which is published by the geometric attitude controller. The callback function is called whenever a new message is received on this topic.
    def _motor_cb(self, msg: Actuators):
        omegas = np.array(msg.velocity, dtype=float)
        if len(omegas) < 4:
            return
        with self._cmd_lock:
            self._omega_cmd     = np.clip(omegas[:4], 0.0, OMEGA_MAX)
            self._last_cmd_time = time.monotonic()

    #This callback just check whether the drone is armed or not by checking the value of rc_aux4 in the SwiftMsgs message. If rc_aux4 is 2000, the drone is considered armed; otherwise, it is disarmed. The armed state is stored in a thread-safe manner using a lock.
    def _drone_cmd_cb(self, msg: SwiftMsgs):
        with self._arm_lock:
            prev = self._armed
            self._armed = (msg.rc_aux4 == 2000)
            if self._armed != prev:
                label = 'ARMED' if self._armed else 'DISARMED'
                self.get_logger().info(f'  >> {label} (rc_aux4={msg.rc_aux4})')

    #This function just checks whether the motor is armed or not
    def is_armed(self) -> bool:
        with self._arm_lock:
            return self._armed

    #This will return the motor conmand along with whether it is freash or a stale command. 
    def get_omega_cmd(self):
        with self._cmd_lock:
            fresh = (time.monotonic() - self._last_cmd_time) < MOTOR_TIMEOUT
            return self._omega_cmd.copy(), fresh

    # ── State / clock ─────────────────────────────────────────────────────────
    #update the state from the MuJoCo simulation. This function is called from the main simulation loop after each physics step. It updates the position, orientation, linear velocity, angular velocity, and simulation time in a thread-safe manner using a lock.
    def update_state(self, pos, quat_mj, vel, ang_vel, sim_time):
        with self._state_lock:
            self._state.update(pos=pos.copy(), quat_mj=quat_mj.copy(),
                               vel=vel.copy(), ang_vel=ang_vel.copy(),
                               sim_time=float(sim_time))


    def publish_clock(self, sim_time: float):
        msg = Clock()
        msg.clock = self._ros_time(sim_time)
        self.clock_pub.publish(msg)

    # ── Odometry (100 Hz, ROS spin thread) ───────────────────────────────────
    #update odomery info from the updated state. 
    def _pub_odometry(self):
        with self._state_lock:
            s = dict(self._state)
        msg = Odometry()
        t = self._ros_time(s['sim_time'])
        msg.header.stamp    = t
        msg.header.frame_id = 'world'
        msg.child_frame_id  = 'swift_pico/base_link'
        msg.pose.pose.position.x = float(s['pos'][0])
        msg.pose.pose.position.y = float(s['pos'][1])
        msg.pose.pose.position.z = float(s['pos'][2])
        q = s['quat_mj']
        msg.pose.pose.orientation = Quaternion(
            x=float(q[1]), y=float(q[2]), z=float(q[3]), w=float(q[0]))
        msg.twist.twist.linear.x  = float(s['vel'][0])
        msg.twist.twist.linear.y  = float(s['vel'][1])
        msg.twist.twist.linear.z  = float(s['vel'][2])
        msg.twist.twist.angular.x = float(s['ang_vel'][0])
        msg.twist.twist.angular.y = float(s['ang_vel'][1])
        msg.twist.twist.angular.z = float(s['ang_vel'][2])
        self.odom_pub.publish(msg)

    # ── Helpers ──────────────────────────────────────────────────────────────
    #camera intrensics and distrotion paramters. 
    @staticmethod
    def _build_camera_info() -> CameraInfo:
        info = CameraInfo()
        info.width  = CAM_W
        info.height = CAM_H
        info.distortion_model = 'plumb_bob'
        info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        info.k = [_FX,  0.0, _CX,
                  0.0,  _FY, _CY,
                  0.0,  0.0, 1.0]
        info.r = [1.0, 0.0, 0.0,
                  0.0, 1.0, 0.0,
                  0.0, 0.0, 1.0]
        info.p = [_FX,  0.0, _CX, 0.0,
                  0.0,  _FY, _CY, 0.0,
                  0.0,  0.0, 1.0, 0.0]
        return info

    @staticmethod
    def _ros_time(sim_time: float) -> Time:
        sec = int(sim_time)
        t   = Time()
        t.sec     = sec
        t.nanosec = int((sim_time - sec) * 1e9)
        return t


# ── Render process (separate OS process, own GIL + EGL context) ─────────────
# Runs the offscreen camera render + /image_raw + /camera_info publish. Kept
# out of the physics/viewer process so it gets real parallel CPU time instead
# of time-sharing one core via the GIL (see module docstring).
def render_process_main(model, shared_state, nq, nv,
                         render_event, render_idle_event, quit_event):
    os.environ.setdefault('MUJOCO_GL', 'egl')

    local_data = mujoco.MjData(model)
    cv_bridge  = CvBridge()

    rclpy.init()
    node = Node('mujoco_camera_bridge')
    reliable = QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        history=HistoryPolicy.KEEP_LAST, depth=1)
    img_pub  = node.create_publisher(RosImage,   '/image_raw',   reliable)
    info_pub = node.create_publisher(CameraInfo, '/camera_info', reliable)
    cam_info = MuJoCoROS2Bridge._build_camera_info()

    debug_frame_path = '/tmp/mujoco_top_cam.png'
    debug_saved = False

    rdr = mujoco.Renderer(model, height=CAM_H, width=CAM_W)
    try:
        while not quit_event.is_set():
            if not render_event.wait(timeout=0.05):
                continue
            render_event.clear()
            render_idle_event.clear()

            with shared_state.get_lock():
                sim_time = shared_state[0]
                local_data.qpos[:] = shared_state[1:1 + nq]
                local_data.qvel[:] = shared_state[1 + nq:1 + nq + nv]
            mujoco.mj_forward(model, local_data)

            rdr.update_scene(local_data, camera='top_cam')
            pixels = rdr.render()
            stamp  = MuJoCoROS2Bridge._ros_time(sim_time)

            img_msg = cv_bridge.cv2_to_imgmsg(pixels, encoding="rgb8")
            img_msg.header.stamp = stamp
            img_msg.header.frame_id = 'camera_optical'
            img_pub.publish(img_msg)

            cam_info.header.stamp = stamp
            cam_info.header.frame_id = 'camera_optical'
            info_pub.publish(cam_info)

            if not debug_saved:
                debug_saved = True
                try:
                    import PIL.Image
                    PIL.Image.fromarray(pixels).save(debug_frame_path)
                except Exception:
                    pass

            render_idle_event.set()
    finally:
        rdr.close()
        node.destroy_node()
        rclpy.shutdown()


# ── Quit-only keyboard ────────────────────────────────────────────────────────
_quit = threading.Event()

def _stdin_reader():
    fd = sys.stdin.fileno()
    try:
        old = termios.tcgetattr(fd)
    except termios.error:
        return
    try:
        tty.setraw(fd)
        while not _quit.is_set():
            if not select.select([sys.stdin], [], [], 0.05)[0]:
                continue
            ch = os.read(fd, 1)
            if ch in (b'\x03', b'q', b'Q', b'\x1b'):
                _quit.set()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _print(msg):
    sys.stdout.write(msg + '\r\n')
    sys.stdout.flush()


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    import os as _os
    _os.environ.setdefault('MUJOCO_GL', 'egl')

    model  = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data   = mujoco.MjData(model)
    motors = MotorDynamics()

    nq, nv = model.nq, model.nv

    # Fork the render process now, before rclpy.init()/the interactive viewer
    # create any DDS participant or GL context in this process -- forking
    # after either exists is unsafe (undefined driver/thread state in the
    # child). shared_state layout: [sim_time, qpos..., qvel...].
    mp_ctx            = mp.get_context('fork')
    shared_state      = mp_ctx.Array('d', 1 + nq + nv)
    render_event      = mp_ctx.Event()
    render_idle_event = mp_ctx.Event()
    render_idle_event.set()
    render_quit_event = mp_ctx.Event()
    render_proc = mp_ctx.Process(
        target=render_process_main,
        args=(model, shared_state, nq, nv,
              render_event, render_idle_event, render_quit_event),
        daemon=True)
    render_proc.start()

    rclpy.init()

    drone_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'drone')

    spin_dofs = [
        model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'spin_m1')],
        model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'spin_m2')],
        model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'spin_m3')],
        model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'spin_m4')],
    ]
    spin_dirs = [+1.0, +1.0, -1.0, -1.0]

    # Spawn at the XML-defined drone pose (model.qpos0 already encodes the
    # <body name="drone" pos="..."> from MODEL_PATH -- e.g. the launch pad
    # location in drone_arena.xml -- so this works for any loaded world
    # without hardcoding x/y or z here; each world's XML places the drone
    # resting on whatever surface -- floor or an elevated pad -- is under
    # it, and a fixed embed offset can't be right for both).
    data.qpos[:] = model.qpos0
    data.qpos[3] = 1.0
    mujoco.mj_forward(model, data)

    render_every = max(1, int(round(1.0 / (SIM_DT * WHYCODE_HZ))))
    render_tick  = 0
    viewer_tick  = 0

    bridge = MuJoCoROS2Bridge()

    # Launch optimized threads
    threading.Thread(target=rclpy.spin,       args=(bridge,), daemon=True).start()
    threading.Thread(target=_stdin_reader,     daemon=True).start()

    last_armed = False
    last_print = time.time()

    def _mj_key_cb(keycode):
        if keycode == 256:
            _quit.set()

    with mujoco.viewer.launch_passive(
            model, data, key_callback=_mj_key_cb) as viewer:

        viewer.cam.distance  = 4.0
        viewer.cam.elevation = -20
        viewer.cam.azimuth   = 135

        _print("")
        _print("=== Swift Pico MuJoCo — ROS2 Bridge ===")
        _print(f"  Camera:  /image_raw  ({CAM_W}×{CAM_H} px, "
               f"hfov={CAM_FOV_RAD:.3f} rad)  — matches the real camera")
        _print(f"  Info:    /camera_info  (fx=fy={_FX:.1f}, cx=cy={_CX:.0f})")
        _print("  Markers:   /whycode_node/markers  (whycode node)")
        _print("  q / Ctrl+C = quit")
        _print("")

        while viewer.is_running() and not _quit.is_set():
            step_start = time.time()
            armed = bridge.is_armed()

            if armed != last_armed:
                if armed:
                    _print("  [ARMED]  motors spinning up ...")
                else:
                    _print("  [DISARMED]  motors off")
                    data.xfrc_applied[drone_body_id] = 0
                    motors.reset()
                last_armed = armed

            if armed:
                omega_cmd, fresh = bridge.get_omega_cmd()
                if not fresh:
                    omega_cmd = motors.omega.copy()
                forces_actual = motors.step(KT * omega_cmd ** 2, SIM_DT)
                data.ctrl[:]  = forces_actual
                motors.apply_disturbances(model, data, drone_body_id)
            else:
                motors.step(np.zeros(4), SIM_DT)
                data.ctrl[:] = 0
                data.xfrc_applied[drone_body_id] = 0

            for i in range(4):
                data.qvel[spin_dofs[i]] = spin_dirs[i] * motors.omega[i]

            mujoco.mj_step(model, data)

            bridge.update_state(
                pos     = data.qpos[0:3],
                quat_mj = data.qpos[3:7],
                vel     = data.qvel[0:3],
                ang_vel = data.qvel[3:6],
                sim_time= data.time,
            )
            bridge.publish_clock(data.time)

            now = time.time()
            if now - last_print >= 1.0:
                pos    = data.qpos[0:3]
                status = "ARMED  " if armed else "DISARMED"
                _print(
                    f"  [{status}]  "
                    f"pos=({pos[0]:+.2f},{pos[1]:+.2f},{pos[2]:+.2f})  "
                    f"rpm={[int(w) for w in motors.omega]}"
                )
                last_print = now

            render_tick += 1
            if render_tick >= render_every and render_idle_event.is_set():
                render_tick = 0
                with shared_state.get_lock():
                    shared_state[0] = data.time
                    shared_state[1:1 + nq] = data.qpos.tolist()
                    shared_state[1 + nq:1 + nq + nv] = data.qvel.tolist()
                render_event.set()

            viewer_tick += 1
            if viewer_tick >= VIEWER_SYNC_EVERY:
                viewer_tick = 0
                viewer.sync()

            elapsed = time.time() - step_start
            if elapsed < SIM_DT:
                time.sleep(SIM_DT - elapsed)

    _quit.set()
    render_quit_event.set()
    render_proc.join(timeout=2.0)
    if render_proc.is_alive():
        render_proc.terminate()
    bridge.destroy_node()
    rclpy.shutdown()
    _print("Bridge stopped.")


if __name__ == '__main__':
    main()
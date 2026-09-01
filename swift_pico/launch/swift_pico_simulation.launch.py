import os

from launch import LaunchDescription
from launch.actions import ExecuteProcess, SetEnvironmentVariable
from launch_ros.actions import Node


def generate_launch_description():

    # Environment variables for MuJoCo bridge to handle large image messages[cite: 2]
    # bridge_env = dict(
    #     os.environ,
    #     MUJOCO_GL='egl',
    #     RMW_FASTRTPS_PUBLICATION_MODE='ASYNCHRONOUS',
    # )

    # ── MuJoCo simulation + ROS2 bridge ──────────────────────────────────────
    mujoco_bridge = ExecuteProcess(
        cmd=['ros2', 'run', 'swift_pico', 'mujoco_bridge'],
        # env=bridge_env,
        output='screen',
    )

    # ── WhyCode marker detection — legacy topics kept for controllers ────────
    whycode = Node(
        package='whycode',
        name='whycode_node',
        executable='whycode_node',
        output='screen',
        parameters=[{
            'img_base_topic': '/image_raw',
            'info_topic': '/camera_info',
            'img_transport': 'raw',
            'circle_diameter': 0.2275,
            'id_bits': 3,
            'id_samples': 720,
            'hamming_dist': 1,
            'num_markers': 1,
            'use_gui': True,
            'min_size': 5,
            'calib_file': '',
            'coords_method': 0,
        }],
        remappings=[
            ('~/markers', '/whycode_node/markers'),
            ('~/processed_image', '/whycode_node/image_out'),
        ],
    )

    roll_pitch_yawrate_thrust_controller = Node(
        package='rotors_control',
        namespace='rotors',
        executable='roll_pitch_yawrate_thrust_controller_node',
        name='roll_pitch_yawrate_thrust_controller',
    )

    swift_interface = Node(
        package='rotors_swift_interface',
        namespace='rotors',
        executable='rotors_swift_interface',
        name='rotors_swift_interface'
    )



    # ── Image view ───────────────────────────────────────────────────────────
    image_view = Node(
        package='image_view',
        executable='image_view',
        namespace='whycode_display',
        name='image_view',
        output='screen',
        remappings=[('image', '/whycode_node/image_out')],
    )

    # Apply asynchronous Fast-DDS publishing to EVERY node in this launch[cite: 2].
    # async_fastdds = SetEnvironmentVariable(
    #     'RMW_FASTRTPS_PUBLICATION_MODE', 'ASYNCHRONOUS')

    return LaunchDescription([
        # async_fastdds,
        mujoco_bridge,
        whycode,
        roll_pitch_yawrate_thrust_controller,
        swift_interface,
        image_view,
    ])
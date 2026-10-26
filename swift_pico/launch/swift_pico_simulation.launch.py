import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import OpaqueFunction
from launch_ros.actions import Node

IMAGE_SHM_NAME = 'swift_pico_image_raw'

def _launch_setup(context, *args, **kwargs):

    fastdds_profile_path = os.path.join(
        get_package_share_directory('swift_pico'), 'config', 'fastdds_profile.xml')

    mujoco_bridge = Node(
        package='swift_pico',
        executable='mujoco_bridge',
        output='screen',
        additional_env={
            'RMW_FASTRTPS_PUBLICATION_MODE': 'ASYNCHRONOUS',
            'FASTRTPS_DEFAULT_PROFILES_FILE': fastdds_profile_path,
        },
    )

    whycode_env = {
        'FASTRTPS_DEFAULT_PROFILES_FILE': fastdds_profile_path,
        'RMW_FASTRTPS_PUBLICATION_MODE': 'ASYNCHRONOUS',
    }

    whycode = Node(
        package='whycode',
        name='whycode_node',
        executable='whycode_node',
        output='screen',
        additional_env=whycode_env,
        parameters=[{
            'img_base_topic': '/image_raw',
            'info_topic': '/camera_info',
            'img_transport': 'raw',
            'img_shm_name': IMAGE_SHM_NAME,
            'img_shm_frame_id': 'camera_optical',
            'circle_diameter': 0.2443,
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

    image_view = Node(
        package='image_view',
        executable='image_view',
        namespace='whycode_display',
        name='image_view',
        output='screen',
        additional_env={'FASTRTPS_DEFAULT_PROFILES_FILE': fastdds_profile_path},
        parameters=[{
            'width': 1000,
            'height': 1000,
        }],
        remappings=[('image', '/whycode_node/image_out')],
    )

    return [
        mujoco_bridge,
        whycode,
        image_view,
    ]

def generate_launch_description():
    return LaunchDescription([
        OpaqueFunction(function=_launch_setup),
    ])
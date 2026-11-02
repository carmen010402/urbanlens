"""
urbanlens_loop.launch.py
Starts Gazebo with the UrbanLens rectangular loop and spawns the Waffle Pi at
the lane's start point (bottom wall, facing +x), from loop_ground_truth.json.
Same pattern as urbanlens_corridor.launch.py, just a different world file and
spawn pose, and with the longer spawn timeout already built in.
Run:
    ros2 launch ~/urban-ground-truth-rover/simulation/urbanlens_loop.launch.py
"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    tb3_launch_dir = os.path.join(get_package_share_directory('turtlebot3_gazebo'), 'launch')
    gazebo_ros_dir = get_package_share_directory('gazebo_ros')
    world = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'urbanlens_loop.world')

    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    x_pose = LaunchConfiguration('x_pose', default='0.0')
    y_pose = LaunchConfiguration('y_pose', default='-1.75')   # bottom-wall lane centre, see loop_ground_truth.json

    gzserver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(gazebo_ros_dir, 'launch', 'gzserver.launch.py')),
        launch_arguments={'world': world}.items())
    gzclient = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(gazebo_ros_dir, 'launch', 'gzclient.launch.py')))
    robot_state_publisher = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(tb3_launch_dir, 'robot_state_publisher.launch.py')),
        launch_arguments={'use_sim_time': use_sim_time}.items())

    model_name = os.environ.get('TURTLEBOT3_MODEL', 'waffle_pi')
    model_sdf = os.path.join(get_package_share_directory('turtlebot3_gazebo'), 'models',
                             'turtlebot3_' + model_name, 'model.sdf')
    spawn_turtlebot = Node(
        package='gazebo_ros', executable='spawn_entity.py', output='screen',
        arguments=['-entity', model_name, '-file', model_sdf,
                   '-x', x_pose, '-y', y_pose, '-z', '0.01', '-Y', '0.0', '-timeout', '120'])

    ld = LaunchDescription()
    for action in (gzserver, gzclient, robot_state_publisher, spawn_turtlebot):
        ld.add_action(action)
    return ld

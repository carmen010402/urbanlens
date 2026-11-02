URBANLENS RECTANGULAR LOOP  (simulation, placeholder scale)
=============================================================
A ring-shaped lane around a solid island, with 5 rooms opening off the outer
wall. S1 is empty (the reference room). S2-S5 each hold one object (table +
chair, trash bag, bed, trolley). Sizes here are a PLACEHOLDER -- edit the
numbers at the top of make_loop_world.py once you know your real / Lego
dimensions, then re-run it.

Files
  make_loop_world.py      generates urbanlens_loop.world + loop_ground_truth.json (edit this to change sizes)
  urbanlens_loop.world    the Gazebo world
  urbanlens_loop.launch.py starts Gazebo with this world, robot at the start point
  loop_ground_truth.json  the manually verified layout (doorway positions, room sizes, which room is the reference)
  loop_layout_preview.png top-view picture

STILL TO DO ONCE YOU'VE SEEN THIS IN GAZEBO
  - resize everything to match your real / Lego plan
  - move S2-S5's objects partway into the doorway path, not just against the back wall,
    so the robot actually has to steer around them
  - build the matching Nav2 map (make_corridor_map.py's approach, extended for this shape)
  - a "room vs lane" and "enter the room, survey, come back out" behaviour for Mission Manager

RUN
  pkill -9 -f gzserver; pkill -9 -f gzclient
  source /usr/share/gazebo-11/setup.sh
  source /opt/ros/humble/setup.bash
  source ~/turtlebot3_ws/install/setup.bash
  export TURTLEBOT3_MODEL=waffle_pi
  ros2 launch ~/urban-ground-truth-rover/simulation/urbanlens_loop.launch.py

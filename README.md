# UrbanLens

A ground-truth rover for corridor change and clearance audits.
Capstone project, Inha University in Tashkent. Target country: Uzbekistan.

The rover drives a corridor, measures the free width of each route segment with LiDAR,
detects objects and damage with a camera, compares the result with the manually verified
layout, and shows everything on a dashboard.

## Hardware and software

| Part | Choice |
|---|---|
| Robot | TurtleBot3 Burger, Raspberry Pi 4B (2 GB), LDS-02 LiDAR |
| OS / middleware | Ubuntu 22.04, ROS 2 Humble |
| Simulation | Gazebo 11, Nav2 |
| Detection | YOLOv8 + BoT-SORT (runs on a laptop) |
| Data | SQLite (`urbanlens.db`) |
| API | Flask, port 8001 |
| Camera server | port 8000 (separate backend of the dashboard teammate) |

## Folders

| Folder | Content |
|---|---|
| `backend/` | Mission Manager, segment database, API, comparison with the layout, model training scripts |
| `backend/detection/` | Detection, BoT-SORT config, position-based duplicate removal |
| `rover-dashboard/` | Dashboard (`index.html`, `style.css`) |
| `simulation/` | Gazebo worlds, launch files, ground-truth layouts, Nav2 map |

## Main files

| File | What it does |
|---|---|
| `backend/mission_manager_node.py` | ROS 2 node. Reads `/odom` and `/scan`, finds the current segment, writes findings. Listens to `/mission_control` (`START` / `STOP`) |
| `backend/segment_manager.py` | SQLite storage, severity, coverage, mission report |
| `backend/api_server.py` | Flask API on port 8001: `/api/coverage`, `/api/findings`, `/api/segment/<id>`, `/api/mission_report` |
| `backend/compare_to_layout.py` | Difference report: measured clearance against the ground-truth layout |
| `backend/detection/position_dedup.py` | Merges repeated detections of the same object by world position |
| `backend/detection/dedup_ros2_node.py` | Runs detection on a recorded video with live `/odom` and `/scan` |
| `backend/detect_combined.py` | Runs the COCO model and the custom model on the same video |
| `backend/train_custom_classes.py` | Trains the custom model (hole, dump, weather_collision) |
| `backend/merge_yolo_datasets.py` | Merges the datasets used for training |
| `simulation/make_loop_world.py` | Generates the rectangular loop world and its ground-truth json |

`backend/server.py` is an old placeholder and is not used.

## Not in this repository

Model weights (`*.pt`), datasets, test videos and `urbanlens.db` are too large and are kept
out by `.gitignore`.
The custom model `best.pt` is needed by `detect_combined.py`. Set `CUSTOM_MODEL_PATH` at
the top of that file to where it is on your machine.

## Setup

```
sudo apt install python3-pip
pip3 install flask flask-cors ultralytics opencv-python
```
ROS 2 Humble, Gazebo 11 and the TurtleBot3 packages (`~/turtlebot3_ws`) must be installed.

## Run the simulation

```
source /usr/share/gazebo-11/setup.sh
source /opt/ros/humble/setup.bash
source ~/turtlebot3_ws/install/setup.bash
export TURTLEBOT3_MODEL=burger
ros2 launch ~/urban-ground-truth-rover/simulation/urbanlens_loop.launch.py
```
Use `urbanlens_corridor.launch.py` for the straight corridor.

## Run a mission

Terminal 1: simulation (above).

Terminal 2: Mission Manager
```
cd ~/urban-ground-truth-rover/backend
source /opt/ros/humble/setup.bash
python3 mission_manager_node.py
```

Terminal 3: start recording, then drive (teleop or Nav2)
```
ros2 topic pub /mission_control std_msgs/String "data: 'START'" --once
ros2 run turtlebot3_teleop teleop_keyboard
```
Send `'STOP'` the same way to pause.

Terminal 4: results
```
cd ~/urban-ground-truth-rover/backend
python3 -c "from segment_manager import print_mission_report; print_mission_report()"
python3 compare_to_layout.py ../simulation/loop_ground_truth.json
```

## Dashboard and API

```
cd ~/urban-ground-truth-rover/backend
python3 api_server.py
```
Test: `curl http://localhost:8001/api/coverage`.
Open `rover-dashboard/index.html` in a browser. The camera view needs the camera server on
port 8000.

## Detection on a recorded video

```
cd ~/urban-ground-truth-rover/backend
python3 detect_combined.py shot1.mp4
```

## Nav2 (autonomous driving in the loop)

The map is `simulation/loop_map.yaml`.
```
ros2 launch turtlebot3_navigation2 navigation2.launch.py use_sim_time:=true map:=$HOME/urban-ground-truth-rover/simulation/loop_map.yaml
```

## Known limits

- Findings repeat about every 2 s while the robot stays in a bad segment.
- STOP does not close the mission.
- Total distance also counts reversing.
- Camera field of view in the dedup code (1.0856 rad) is a simulation value.
- Severity thresholds are provisional.
- The simulation camera does not render in WSL, so detection runs on recorded video.
- The real camera is not mounted yet.

## Team

| Name | ID | Part |
|---|---|---|
| Chaw Khin Su | 12230339 | Engineering |
| Jamolov Dilmurod | 12230271 | Engineering |
| Ahmadjonov Asadbek | 12230327 | Testing and QA |

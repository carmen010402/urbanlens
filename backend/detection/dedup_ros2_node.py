"""
dedup_ros2_node.py

Integration test: combines your REAL, LIVE /odom and /scan from Gazebo
with YOLO+BoT-SORT detection run on one of your RECORDED test videos
(shot1.mp4, shot2.mp4, short3.mp4), and feeds everything through the
position_dedup.py logic.

WHY A RECORDED VIDEO INSTEAD OF GAZEBO'S CAMERA:
Gazebo's camera sensor needs GPU-accelerated rendering (OGRE), which your
WSL setup doesn't have (confirmed: /odom and /scan work perfectly, but no
camera topic is ever published, no error, it just silently never starts).
So this node drives the DETECTION half from your recorded footage, while
pulling the POSE and LIDAR half from the real, live simulation. The bearing
+ range + pose math is identical to what will run on the real robot later --
only the source of the video frames changes.

HOW TO RUN:
1. Make sure Gazebo is already running in another terminal:
     ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py headless:=True gui:=false
2. In a new terminal, source ROS2 + your workspace, then:
     python3 dedup_ros2_node.py shot1.mp4
3. Optionally drive the robot around in a third terminal while this runs:
     ros2 run turtlebot3_teleop teleop_keyboard
   (moving the simulated robot changes /odom, which changes the world
   positions computed for each detection -- try it and watch the output)

This is a plain rclpy node -- it does not need to live inside a ROS2
package to run, as long as ROS2 and your workspace are sourced first.
"""

import sys
import math
import json

import cv2
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from ultralytics import YOLO

from position_dedup import (
    PositionRegistry,
    bbox_center_to_bearing,
    lidar_range_at_bearing,
    estimate_world_position,
    quaternion_to_yaw,
    CAMERA_HORIZONTAL_FOV_RAD,
)

PERSON_CLASS = 0
RELEVANT_OBSTACLE_CLASSES = {56, 24, 28, 26, 13, 58, 41}  # chair, backpack, suitcase, handbag, bench, potted plant, cup
MIN_HITS_TO_COUNT = 5


class DedupNode(Node):
    def __init__(self, video_path: str):
        super().__init__('position_dedup_test_node')

        # --- live ROS2 state, updated by subscriptions as messages arrive ---
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_heading = 0.0
        self.have_odom = False

        self.scan_ranges = []
        self.scan_angle_min = 0.0
        self.scan_angle_increment = 0.0
        self.have_scan = False

        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

        # --- detection pipeline (same as detect.py, minus display) ---
        self.model = YOLO('yolov8s.pt')
        self.cap = cv2.VideoCapture(video_path)
        self.frame_num = 0
        self.track_hit_counts = {}
        self.person_registry = PositionRegistry(dedup_radius=0.6)
        self.obstacle_registry = PositionRegistry(dedup_radius=0.6)

        # Process one video frame every 100ms, keeping this a normal ROS2
        # timer callback so /odom and /scan keep updating in between.
        self.create_timer(0.1, self.process_next_frame)

        self.get_logger().info(f"Started. Waiting for /odom and /scan before processing '{video_path}'...")

    def odom_callback(self, msg: Odometry):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.robot_heading = quaternion_to_yaw(q.x, q.y, q.z, q.w)
        self.have_odom = True

    def scan_callback(self, msg: LaserScan):
        self.scan_ranges = msg.ranges
        self.scan_angle_min = msg.angle_min
        self.scan_angle_increment = msg.angle_increment
        self.have_scan = True

    def process_next_frame(self):
        if not (self.have_odom and self.have_scan):
            self.get_logger().info("Still waiting for /odom and /scan...")
            return

        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().info("Video finished. Writing results...")
            self.write_results()
            rclpy.shutdown()
            return

        self.frame_num += 1
        frame_width = frame.shape[1]

        results = self.model.track(
            frame, persist=True, conf=0.35,
            tracker="botsort.yaml", classes=None, verbose=False,
        )

        if results[0].boxes.id is None:
            return

        for box, cls_id, track_id in zip(results[0].boxes.xyxy,
                                          results[0].boxes.cls,
                                          results[0].boxes.id):
            cls_id = int(cls_id)
            track_id = int(track_id)
            x1, y1, x2, y2 = box.tolist()
            center_x = (x1 + x2) / 2

            self.track_hit_counts[track_id] = self.track_hit_counts.get(track_id, 0) + 1
            if self.track_hit_counts[track_id] < MIN_HITS_TO_COUNT:
                continue

            if cls_id != PERSON_CLASS and cls_id not in RELEVANT_OBSTACLE_CLASSES:
                continue

            # --- the actual fix: convert this detection to a world position ---
            bearing = bbox_center_to_bearing(center_x, frame_width, CAMERA_HORIZONTAL_FOV_RAD)
            distance = lidar_range_at_bearing(
                self.scan_ranges, self.scan_angle_min, self.scan_angle_increment, bearing
            )
            if distance is None:
                # No valid LiDAR reading in that direction (out of range /
                # sensor gap) -- can't place it in the world, skip for now.
                continue

            world_x, world_y = estimate_world_position(
                self.robot_x, self.robot_y, self.robot_heading, bearing, distance
            )

            obj_type = "person" if cls_id == PERSON_CLASS else self.model.names[cls_id]
            registry = self.person_registry if cls_id == PERSON_CLASS else self.obstacle_registry
            result = registry.add_or_update(obj_type, world_x, world_y, self.frame_num,
                                             extra={"track_id": track_id})
            status = "NEW" if result.sightings == 1 else f"MATCHED #{result.id}"
            self.get_logger().info(
                f"Frame {self.frame_num}: {obj_type} (track_id {track_id}) "
                f"-> world ({world_x:.2f},{world_y:.2f}) [{status}]"
            )

    def write_results(self):
        output = {
            "unique_people_count": len(self.person_registry.objects),
            "unique_obstacle_count": len(self.obstacle_registry.objects),
            "people": [o.__dict__ for o in self.person_registry.objects],
            "obstacles": [o.__dict__ for o in self.obstacle_registry.objects],
        }
        with open("dedup_results.json", "w") as f:
            json.dump(output, f, indent=2)
        print("\n=== FINAL RESULTS (position-based dedup) ===")
        print(f"Unique people: {len(self.person_registry.objects)}")
        print(f"Unique obstacles: {len(self.obstacle_registry.objects)}")
        print("Saved to dedup_results.json")


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 dedup_ros2_node.py <video_file.mp4>")
        sys.exit(1)

    rclpy.init()
    node = DedupNode(sys.argv[1])
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

"""
mission_manager_node.py

This is the "Mission Manager Node" from your architecture diagram. It's the
real glue between live ROS2 sensor data and segment_manager.py -- no fake
demo numbers, no mismatched video+sim data like our earlier test. Everything
here comes from the SAME live moment (Gazebo simulation now, real robot
later), so the numbers are honest.

WHAT IT DOES:
1. Subscribes to /odom -> works out WHERE the robot is along the route
   (measured from the spot where you pressed START, in the direction the robot
   was facing then) -> that decides the segment (S1, S2, S3...). Driving
   backward therefore goes back to the earlier segment. The total distance
   driven is counted separately (it always goes up) for the mission report.
2. Subscribes to /scan -> computes "actual clearance" at the robot's
   current position (distance to obstacles on left + right side).
3. Every couple of seconds, calls segment_manager.record_finding() with
   REAL data for whatever segment the robot is currently in.
4. Listens for simple START/STOP commands on a topic.

WHAT'S STILL MISSING (by design, until hardware exists):
Object TYPE (chair, box, person...) normally comes from detect.py's camera
detection. Since Gazebo's camera can't render in your WSL setup, this node
uses a generic placeholder ("Obstruction (LiDAR only)") whenever clearance
drops below reference. Once a real camera feed exists, replace the
placeholder with detect.py's actual detected object_type -- everything
else here stays the same.

HOW TO RUN:
Terminal 1 (keep running):
  ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py headless:=True gui:=false

Terminal 2:
  source /opt/ros/humble/setup.bash
  source ~/turtlebot3_ws/install/setup.bash
  python3 mission_manager_node.py

Terminal 3 (drive the robot around so segments actually change):
  ros2 run turtlebot3_teleop teleop_keyboard

Terminal 4 (anytime, to check results):
  python3 -c "from segment_manager import print_mission_report; print_mission_report()"
"""

import math
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String

from segment_manager import init_db, define_segments, record_finding, update_mission_progress

# --- Configuration: adjust these to match your real corridor later ---
SEGMENT_COUNT = 5
SEGMENT_LENGTH_M = 2.0          # each segment = this many meters ALONG the route
REFERENCE_CLEARANCE_M = 1.50    # design/expected corridor width
UPDATE_INTERVAL_SEC = 2.0       # how often to record a finding


def quaternion_to_yaw(qx, qy, qz, qw):
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def progress_along_route(x, y, origin_x, origin_y, origin_yaw):
    """Metres travelled from the route origin, measured ALONG the direction the
    robot faced at START. Going forward increases it, backing up decreases it,
    and sideways movement does not change it."""
    return (x - origin_x) * math.cos(origin_yaw) + (y - origin_y) * math.sin(origin_yaw)


def segment_index_for_progress(progress_m, segment_length=SEGMENT_LENGTH_M, segment_count=SEGMENT_COUNT):
    """0-based segment number for a position along the route (clamped to the first / last segment)."""
    return max(0, min(int(progress_m // segment_length), segment_count - 1))


class MissionManagerNode(Node):
    def __init__(self):
        super().__init__('mission_manager_node')

        init_db(reset=True)
        define_segments({f"S{i+1}": REFERENCE_CLEARANCE_M for i in range(SEGMENT_COUNT)})
        self.get_logger().info(f"Defined {SEGMENT_COUNT} segments, {SEGMENT_LENGTH_M}m each.")

        self.mission_running = False
        self.have_odom = False
        self.origin = None              # (x, y, yaw) of the robot when START was first pressed
        self.progress = 0.0             # metres along the route from that origin
        self.total_distance = 0.0       # total path driven (only ever increases)
        self.last_x = None
        self.last_y = None
        self.current_segment_index = 0

        self.scan_ranges = []
        self.scan_angle_min = 0.0
        self.scan_angle_increment = 0.0
        self.range_max = 3.5

        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_heading = 0.0

        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.create_subscription(String, '/mission_control', self.control_callback, 10)

        self.create_timer(UPDATE_INTERVAL_SEC, self.record_current_state)

        self.get_logger().info(
            "Mission Manager ready. Publish 'START' to /mission_control to begin recording:\n"
            "  ros2 topic pub /mission_control std_msgs/String \"data: 'START'\" --once"
        )

    def control_callback(self, msg: String):
        command = msg.data.strip().upper()
        if command == "START":
            if not self.have_odom:
                self.get_logger().info("No /odom yet - is the simulator running? Press START again in a moment.")
                return
            if self.origin is None:
                # The route origin is fixed by the FIRST start: press START at the beginning of
                # the route, facing along it. STOP then START later just pauses / resumes.
                self.origin = (self.robot_x, self.robot_y, self.robot_heading)
                self.get_logger().info(
                    f"Route origin set at x={self.robot_x:.2f} y={self.robot_y:.2f} "
                    f"heading={math.degrees(self.robot_heading):.0f} deg")
            self.mission_running = True
            self.get_logger().info(">>> MISSION STARTED <<<")
        elif command == "STOP":
            self.mission_running = False
            self.get_logger().info(">>> MISSION STOPPED <<<")

    def odom_callback(self, msg: Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.robot_heading = quaternion_to_yaw(q.x, q.y, q.z, q.w)
        self.robot_x = x
        self.robot_y = y
        self.have_odom = True

        if self.last_x is not None and self.mission_running:
            step = math.hypot(x - self.last_x, y - self.last_y)
            self.total_distance += step

        self.last_x = x
        self.last_y = y

        # Which segment are we in? Based on POSITION along the route, so reversing
        # goes back to the earlier segment (total_distance above is only for the report).
        if self.origin is not None:
            ox, oy, oyaw = self.origin
            self.progress = progress_along_route(x, y, ox, oy, oyaw)
            self.current_segment_index = segment_index_for_progress(self.progress)

    def scan_callback(self, msg: LaserScan):
        self.scan_ranges = msg.ranges
        self.scan_angle_min = msg.angle_min
        self.scan_angle_increment = msg.angle_increment
        self.range_max = msg.range_max

    def _range_at_relative_angle(self, relative_angle_rad: float):
        """Look up the LiDAR's distance reading at an angle relative to the
        robot's own forward direction (e.g. +pi/2 = directly to its left)."""
        if not self.scan_ranges:
            return None
        two_pi = 2 * math.pi
        target = relative_angle_rad % two_pi
        index = round((target - self.scan_angle_min) / self.scan_angle_increment)
        if index < 0 or index >= len(self.scan_ranges):
            return None
        r = self.scan_ranges[index]
        if r == float("inf") or r != r:
            return self.range_max  # nothing detected -> treat as "clear to sensor's max range"
        return r

    def compute_actual_clearance(self) -> float:
        """Approximates corridor width at the robot's current position:
        distance to the left wall/obstacle + distance to the right one."""
        left = self._range_at_relative_angle(math.pi / 2)     # 90 degrees left
        right = self._range_at_relative_angle(-math.pi / 2)   # 90 degrees right (i.e. 270)
        if left is None or right is None:
            return REFERENCE_CLEARANCE_M  # no data yet, assume clear
        return round(left + right, 3)

    def record_current_state(self):
        if not self.mission_running:
            return
        if not self.scan_ranges:
            self.get_logger().info("Waiting for /scan data...")
            return

        segment_id = f"S{self.current_segment_index + 1}"
        actual_clearance = self.compute_actual_clearance()

        # No live camera in this simulation environment -- see file header.
        # Once a real camera exists, pass detect.py's real object_type here instead.
        object_type = None
        if actual_clearance < REFERENCE_CLEARANCE_M - 0.05:
            object_type = "Obstruction (LiDAR only, no camera)"

        result = record_finding(
            segment_id=segment_id,
            object_type=object_type,
            actual_clearance_m=actual_clearance,
            lidar_distance_m=actual_clearance,
            robot_x=self.robot_x, robot_y=self.robot_y,
            robot_orientation_deg=math.degrees(self.robot_heading),
        )

        status = f"FINDING ({result['finding_type']}, sev {result['severity']}/5)" if result["has_finding"] else "clear"
        self.get_logger().info(
            f"[{segment_id}] position={self.progress:.2f}m distance={self.total_distance:.2f}m "
            f"clearance={actual_clearance:.2f}m -> {status}"
        )

        # THE FIX: actually save the distance traveled so far into the
        # mission table, so generate_mission_report() reports it correctly
        # instead of showing 0.0m.
        update_mission_progress(distance_m=round(self.total_distance, 2), status="RUNNING")


def main():
    rclpy.init()
    node = MissionManagerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
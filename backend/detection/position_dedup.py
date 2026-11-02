"""
position_dedup.py

Deduplicates detections (people, chairs, boxes, etc.) using their estimated
REAL-WORLD position, instead of relying on the vision tracker's track_id.

WHY THIS EXISTS:
Track-ID-based dedup (BoT-SORT, ByteTrack) breaks whenever the camera pans,
an object is briefly occluded, or something leaves and re-enters the frame --
all of which happen constantly during a 180-degree rover sweep. We proved
this on shot1/2/3.mp4: the same chair/person got 2-5 different track_ids.

This module fixes that by converting each detection into a world coordinate
(using the robot's pose + LiDAR range at that bearing), then checking:
"is there already something of this type within DEDUP_RADIUS meters?"
If yes -> it's the same object, just update it. If no -> it's genuinely new.

This has NO dependency on ROS2, YOLO, or OpenCV -- it's pure geometry and
bookkeeping, so you can read/test it on its own. dedup_ros2_node.py (the
other file) wires this up to real /odom, /scan, and camera frames.
"""

import math
import json
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Tunable parameters
# ---------------------------------------------------------------------------

# If a new detection of the SAME type is found within this many meters of an
# already-known object, we treat it as the same physical object, not a new one.
# Tune this: too small -> same object counted twice; too large -> two real
# objects standing close together get merged into one.
DEDUP_RADIUS_METERS = 0.6

# The TurtleBot3 Waffle Pi's camera horizontal field of view, taken directly
# from its Gazebo model.sdf (<horizontal_fov>1.085595</horizontal_fov>).
# This is needed to convert "where in the image" into "what angle from the
# robot's facing direction".
CAMERA_HORIZONTAL_FOV_RAD = 1.085595


@dataclass
class WorldObject:
    """A single real-world object we've detected and are tracking by position."""
    id: int
    obj_type: str
    x: float
    y: float
    first_seen_frame: int
    last_seen_frame: int
    sightings: int = 1
    # Keep the highest-confidence sighting's metadata for reporting later
    extra: dict = field(default_factory=dict)


class PositionRegistry:
    """
    Keeps track of every real-world object seen so far, and decides whether
    a new detection is something already known or genuinely new.
    """

    def __init__(self, dedup_radius: float = DEDUP_RADIUS_METERS):
        self.dedup_radius = dedup_radius
        self.objects: list[WorldObject] = []
        self._next_id = 1

    def _distance(self, x1, y1, x2, y2) -> float:
        return math.hypot(x2 - x1, y2 - y1)

    def add_or_update(self, obj_type: str, x: float, y: float, frame_num: int,
                       extra: Optional[dict] = None) -> WorldObject:
        """
        Call this once per accepted detection. Returns the WorldObject it
        matched to (existing) or created (new) -- check `.sightings == 1`
        to tell which happened.
        """
        # Only compare against objects of the SAME type -- a chair and a
        # person standing in the same spot are still two different things.
        candidates = [o for o in self.objects if o.obj_type == obj_type]

        best_match = None
        best_dist = self.dedup_radius
        for obj in candidates:
            d = self._distance(obj.x, obj.y, x, y)
            if d <= best_dist:
                best_match = obj
                best_dist = d

        if best_match is not None:
            # Same physical object -- update its position slightly (objects
            # can be static but our position *estimate* has noise) and mark
            # it as seen again instead of creating a duplicate.
            best_match.x = (best_match.x + x) / 2  # simple running average
            best_match.y = (best_match.y + y) / 2
            best_match.last_seen_frame = frame_num
            best_match.sightings += 1
            return best_match

        # No match close enough -- this is a genuinely new object.
        new_obj = WorldObject(
            id=self._next_id,
            obj_type=obj_type,
            x=x, y=y,
            first_seen_frame=frame_num,
            last_seen_frame=frame_num,
            extra=extra or {},
        )
        self._next_id += 1
        self.objects.append(new_obj)
        return new_obj

    def to_json(self) -> str:
        return json.dumps([o.__dict__ for o in self.objects], indent=2)


# ---------------------------------------------------------------------------
# Geometry: turning a detection into a world position
# ---------------------------------------------------------------------------

def bbox_center_to_bearing(center_x: float, frame_width: int,
                            camera_fov_rad: float = CAMERA_HORIZONTAL_FOV_RAD) -> float:
    """
    Converts a detection's horizontal pixel position into an angle (radians)
    relative to the CENTER of the camera's view.
    Negative = left of center, positive = right of center, 0 = dead ahead.
    """
    # Normalize center_x to a -0.5..+0.5 range across the frame width
    normalized = (center_x / frame_width) - 0.5
    return normalized * camera_fov_rad


def lidar_range_at_bearing(ranges: list[float], angle_min: float,
                            angle_increment: float, bearing_rad: float) -> Optional[float]:
    """
    Given a full LiDAR scan (as published on /scan) and a bearing (radians,
    relative to the robot's own forward direction), returns the LiDAR's
    measured distance in that direction, or None if that reading is invalid
    (inf / out of range -- common when nothing is in range, or a sensor gap).
    """
    # /scan angles are relative to the robot's own frame, going counter-
    # clockwise from angle_min. Wrap into [0, 2*pi) to be safe.
    two_pi = 2 * math.pi
    target_angle = bearing_rad % two_pi
    index = round((target_angle - angle_min) / angle_increment)

    if index < 0 or index >= len(ranges):
        return None

    r = ranges[index]
    if r == float("inf") or r != r:  # inf or NaN
        return None
    return r


def estimate_world_position(robot_x: float, robot_y: float, robot_heading_rad: float,
                             bearing_from_robot_rad: float, distance_m: float) -> tuple[float, float]:
    """
    Combines the robot's own pose (from /odom) with a bearing+distance
    (from camera+LiDAR fusion) to get the object's position in the world
    (odom) frame -- the same frame /odom itself is published in.
    """
    absolute_angle = robot_heading_rad + bearing_from_robot_rad
    obj_x = robot_x + distance_m * math.cos(absolute_angle)
    obj_y = robot_y + distance_m * math.sin(absolute_angle)
    return obj_x, obj_y


def quaternion_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    """
    /odom gives orientation as a quaternion. We only care about yaw
    (rotation around the vertical axis) since the robot drives on a flat
    floor. Standard quaternion-to-yaw formula.
    """
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


# ---------------------------------------------------------------------------
# Self-contained demo -- run this file directly to see it work on fake data
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Demo: simulating the exact shot1.mp4 failure case, but with dedup fixed.\n")

    registry = PositionRegistry(dedup_radius=0.6)

    # Simulate a robot panning in place (heading changes, position fixed),
    # "seeing" the same couch 3 times at different pan angles -- exactly
    # like track_ids 1, 47, 65 did in your real shot1.mp4 results.
    robot_x, robot_y = 0.0, 0.0

    sightings = [
        # (frame_num, heading_deg, bearing_in_frame_deg, lidar_distance_m)
        (5,   10, -5,  2.0),   # couch seen while panning left
        (310, 12, -3,  2.05),  # "lost" then re-seen -- would have been a NEW track_id
        (327, 11, -4,  1.98),  # seen again -- another would-be new track_id
    ]

    for frame_num, heading_deg, bearing_deg, dist in sightings:
        heading_rad = math.radians(heading_deg)
        bearing_rad = math.radians(bearing_deg)
        x, y = estimate_world_position(robot_x, robot_y, heading_rad, bearing_rad, dist)
        result = registry.add_or_update("bench", x, y, frame_num)
        status = "NEW object" if result.sightings == 1 else f"MATCHED existing object #{result.id}"
        print(f"Frame {frame_num}: detected 'bench' at world ({x:.2f}, {y:.2f}) -> {status}")

    print(f"\nFinal object count: {len(registry.objects)} (should be 1, not 3)")
    print("\nRegistry contents:")
    print(registry.to_json())

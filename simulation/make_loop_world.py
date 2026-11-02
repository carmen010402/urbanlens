"""
make_loop_world.py

Builds the UrbanLens RECTANGULAR LOOP for Gazebo: a ring-shaped lane around a
solid island, with 5 rooms (segments) opening off the outer wall at chosen
points -- S1 is empty (the reference room), S2-S5 each hold one object.

Geometry, top view (not to scale):

              S1 (empty, reference)
         _____[door]_____
        |                |
  S4 ->[door]   island   [door]<- S2
        |________________|
             [door]
               S3

(S5 also sits on the top wall, further round -- see ROOMS below)

Run:  python3 make_loop_world.py
Writes: urbanlens_loop.world, loop_ground_truth.json, loop_layout_preview.png
"""
import json
import math
import os

# ---- scale knobs -- change these once you know the real/Lego dimensions ----
LANE_WIDTH = 0.75            # drivable width of the ring lane
ISLAND_HALF_X, ISLAND_HALF_Y = 1.5, 1.0     # half-size of the solid centre block
WALL_T, WALL_H = 0.10, 1.0
DOORWAY_WIDTH = 0.60         # opening cut into the outer wall for each room

OUTER_HALF_X = ISLAND_HALF_X + LANE_WIDTH
OUTER_HALF_Y = ISLAND_HALF_Y + LANE_WIDTH

# Each room: which outer wall it opens off, position along that wall (0=one
# corner's end, 1=the other end, kept away from the corners), which way it
# sticks (always outward, away from the island), its size, and its one object
# (None for the empty reference room). Object offset is measured from the
# doorway, going INTO the room, in (along-wall, into-room) room-local metres.
ROOMS = [
    dict(name="S1", wall="top",    t=0.28, width=1.0, depth=1.0, color=(0.75,0.90,0.75),
         object=None),
    dict(name="S2", wall="right",  t=0.50, width=1.0, depth=1.2, color=(0.95,0.80,0.45),
         object=dict(kind="table_chair", along=0.0, into=0.75)),
    dict(name="S3", wall="bottom", t=0.50, width=0.7, depth=0.7, color=(0.95,0.80,0.45),
         object=dict(kind="trash_bag", along=0.0, into=0.40)),
    dict(name="S4", wall="left",   t=0.50, width=1.4, depth=1.1, color=(0.95,0.80,0.45),
         object=dict(kind="bed", along=0.0, into=0.65)),
    dict(name="S5", wall="top",    t=0.78, width=0.9, depth=0.9, color=(0.95,0.80,0.45),
         object=dict(kind="trolley", along=0.0, into=0.45)),
]

HERE = os.path.dirname(os.path.abspath(__file__))


def box_model(name, x, y, z, sx, sy, sz, color, yaw=0.0, collision=True, static=True):
    r, g, b = color
    col = (f"""
      <collision name="collision"><geometry><box><size>{sx} {sy} {sz}</size></box></geometry></collision>"""
           if collision else "")
    return f"""
    <model name="{name}">
      <static>{"true" if static else "false"}</static>
      <pose>{x:.4f} {y:.4f} {z:.4f} 0 0 {yaw:.6f}</pose>
      <link name="link">{col}
        <visual name="visual">
          <geometry><box><size>{sx} {sy} {sz}</size></box></geometry>
          <material><ambient>{r} {g} {b} 1</ambient><diffuse>{r} {g} {b} 1</diffuse></material>
        </visual>
      </link>
    </model>"""


def wall_segments_with_gaps(name_prefix, axis, fixed_coord, span_lo, span_hi, thickness, gaps, color):
    """
    One straight wall (full length span_lo..span_hi along `axis`, 'x' or 'y',
    sitting at `fixed_coord` on the other axis), with rectangular `gaps`
    (list of (centre, width) along the span) cut out of it for doorways.
    Returns a list of box_model(...) SDF strings, one per remaining segment.
    """
    cuts = sorted(gaps)
    pieces = []
    cursor = span_lo
    for center, width in cuts:
        g0, g1 = center - width / 2, center + width / 2
        if g0 > cursor:
            pieces.append((cursor, g0))
        cursor = max(cursor, g1)
    if cursor < span_hi:
        pieces.append((cursor, span_hi))

    out = []
    for i, (a, b) in enumerate(pieces):
        length = b - a
        mid = (a + b) / 2
        if length <= 1e-6:
            continue
        if axis == "x":
            out.append(box_model(f"{name_prefix}_{i}", mid, fixed_coord, WALL_H / 2, length, thickness, WALL_H, color))
        else:
            out.append(box_model(f"{name_prefix}_{i}", fixed_coord, mid, WALL_H / 2, thickness, length, WALL_H, color))
    return out


def wall_point_and_normal(wall, t):
    """Point ON THE OUTER WALL CENTRELINE at fraction t along it, and the unit
    vector pointing outward (away from the island)."""
    ux, uy = OUTER_HALF_X, OUTER_HALF_Y
    if wall == "top":    return (-ux + 2 * ux * t, uy), (0, 1)
    if wall == "bottom": return (-ux + 2 * ux * t, -uy), (0, -1)
    if wall == "left":   return (-ux, -uy + 2 * uy * t), (-1, 0)
    if wall == "right":  return (ux, -uy + 2 * uy * t), (1, 0)
    raise ValueError(wall)


def room_model_parts(room):
    (wx, wy), (nx, ny) = wall_point_and_normal(room["wall"], room["t"])
    yaw = math.atan2(ny, nx) - math.pi / 2     # so the room's "width" axis runs ALONG the wall
    # centre of the room, offset outward from the wall by half its depth (+ wall thickness)
    cx = wx + nx * (WALL_T + room["depth"] / 2)
    cy = wy + ny * (WALL_T + room["depth"] / 2)

    w, d = room["width"], room["depth"]
    parts = []
    # three walls: back (opposite the doorway) + two sides; local frame: local-x = along width, local-y = along depth (0 at doorway, +d at the back)
    def local_to_world(lx, ly):
        return (cx + lx * math.cos(yaw) - ly * math.sin(yaw),
                cy + lx * math.sin(yaw) + ly * math.cos(yaw))

    # back wall (full width), at local y = +d/2 (far from the doorway)
    bx, by = local_to_world(0, d / 2)
    parts.append(box_model(f"{room['name']}_back", bx, by, WALL_H / 2, w + 2 * WALL_T, WALL_T, WALL_H, (0.8, 0.8, 0.82), yaw=yaw))
    # two side walls, each spanning the room's full depth
    for side in (-1, 1):
        sx, sy = local_to_world(side * (w / 2 + WALL_T / 2), 0)
        parts.append(box_model(f"{room['name']}_side{side}", sx, sy, WALL_H / 2, WALL_T, d + WALL_T, WALL_H, (0.8, 0.8, 0.82), yaw=yaw))
    # floor tint, visual only (helps tell rooms apart in the GUI)
    fx, fy = local_to_world(0, d / 2)
    parts.append(box_model(f"{room['name']}_floor", fx, fy, 0.002, w, d, 0.004, room["color"], yaw=yaw, collision=False))

    # the one object inside, if any (simple stand-in boxes, same style as the straight-corridor test)
    OBJ_SIZE = {
        "table_chair": (0.5, 0.5, 0.45), "trash_bag": (0.35, 0.35, 0.35),
        "bed": (1.0, 0.55, 0.45), "trolley": (0.5, 0.35, 0.6),
    }
    OBJ_COLOR = {
        "table_chair": (0.55, 0.35, 0.15), "trash_bag": (0.25, 0.45, 0.2),
        "bed": (0.65, 0.2, 0.25), "trolley": (0.3, 0.3, 0.35),
    }
    if room["object"]:
        o = room["object"]
        ow, od, oh = OBJ_SIZE[o["kind"]]
        ox, oy = local_to_world(o["along"], o["into"])
        parts.append(box_model(f"{room['name']}_object_{o['kind']}", ox, oy, oh / 2, ow, od, oh, OBJ_COLOR[o["kind"]], yaw=yaw))

    doorway = dict(x=wx, y=wy, yaw=yaw, width=DOORWAY_WIDTH)
    return parts, doorway


def build_world():
    parts = []
    grey = (0.78, 0.78, 0.82)

    doorways_by_wall = {"top": [], "bottom": [], "left": [], "right": []}
    room_parts_all = []
    for r in ROOMS:
        rp, door = room_model_parts(r)
        room_parts_all += rp
        axis_coord = door["x"] if r["wall"] in ("top", "bottom") else door["y"]
        doorways_by_wall[r["wall"]].append((axis_coord, door["width"]))

    # outer ring wall, one straight piece per side, each with its rooms' doorway gaps cut in
    ux, uy = OUTER_HALF_X, OUTER_HALF_Y
    parts += wall_segments_with_gaps("wall_top",    "x", uy,  -ux - WALL_T, ux + WALL_T, WALL_T, doorways_by_wall["top"], grey)
    parts += wall_segments_with_gaps("wall_bottom", "x", -uy, -ux - WALL_T, ux + WALL_T, WALL_T, doorways_by_wall["bottom"], grey)
    parts += wall_segments_with_gaps("wall_left",   "y", -ux, -uy, uy, WALL_T, doorways_by_wall["left"], grey)
    parts += wall_segments_with_gaps("wall_right",  "y", ux,  -uy, uy, WALL_T, doorways_by_wall["right"], grey)

    # the solid island in the middle (no gaps -- nothing opens onto it in this version)
    parts.append(box_model("island", 0, 0, WALL_H / 2, 2 * ISLAND_HALF_X, 2 * ISLAND_HALF_Y, WALL_H, (0.65, 0.65, 0.7)))

    # lane floor tint (visual only)
    parts.append(box_model("lane_floor", 0, 0, 0.001, 2 * (ux + WALL_T), 2 * (uy + WALL_T), 0.002, (0.80, 0.88, 0.95), collision=False))

    parts += room_parts_all

    return f"""<?xml version="1.0" ?>
<sdf version="1.6">
  <world name="urbanlens_loop">
    <scene><ambient>0.55 0.55 0.55 1</ambient><background>0.75 0.82 0.9 1</background><shadows>true</shadows></scene>
    <light name="sun" type="directional">
      <cast_shadows>true</cast_shadows><pose>0 0 10 0 0 0</pose>
      <diffuse>0.85 0.85 0.85 1</diffuse><specular>0.2 0.2 0.2 1</specular>
      <attenuation><range>1000</range><constant>0.9</constant><linear>0.01</linear><quadratic>0.001</quadratic></attenuation>
      <direction>-0.4 0.2 -0.9</direction>
    </light>
    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision"><geometry><plane><normal>0 0 1</normal><size>60 60</size></plane></geometry>
          <surface><friction><ode><mu>100</mu><mu2>50</mu2></ode></friction></surface></collision>
        <visual name="visual"><cast_shadows>false</cast_shadows>
          <geometry><plane><normal>0 0 1</normal><size>60 60</size></plane></geometry>
          <material><ambient>0.4 0.4 0.4 1</ambient><diffuse>0.4 0.4 0.4 1</diffuse></material></visual>
      </link>
    </model>
{''.join(parts)}
  </world>
</sdf>
"""


def build_ground_truth():
    segs = {}
    for r in ROOMS:
        (wx, wy), (nx, ny) = wall_point_and_normal(r["wall"], r["t"])
        segs[r["name"]] = {
            "wall": r["wall"], "doorway_center": [round(wx, 3), round(wy, 3)],
            "doorway_width_m": DOORWAY_WIDTH, "room_width_m": r["width"], "room_depth_m": r["depth"],
            "is_reference": r["object"] is None,
            "object": r["object"]["kind"] if r["object"] else None,
        }
    return {
        "description": "Manually verified layout of the simulated UrbanLens rectangular loop",
        "lane_width_m": LANE_WIDTH,
        "island_half_extent_m": [ISLAND_HALF_X, ISLAND_HALF_Y],
        "outer_half_extent_m": [OUTER_HALF_X, OUTER_HALF_Y],
        "robot_spawn": {"x": 0.0, "y": -OUTER_HALF_Y + LANE_WIDTH / 2, "facing": "+x (along the bottom wall)"},
        "segments": segs,
    }


def draw_preview(gt):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import numpy as np

    fig, ax = plt.subplots(figsize=(10, 8.5), dpi=150)
    ux, uy = OUTER_HALF_X, OUTER_HALF_Y
    ax.add_patch(Rectangle((-ux - WALL_T, -uy - WALL_T), 2 * (ux + WALL_T), 2 * (uy + WALL_T), fc="none", ec="#444", lw=2))
    ax.add_patch(Rectangle((-ISLAND_HALF_X, -ISLAND_HALF_Y), 2 * ISLAND_HALF_X, 2 * ISLAND_HALF_Y, fc="#ccccd2", ec="#444", lw=2))
    mx, my = ux - LANE_WIDTH / 2, uy - LANE_WIDTH / 2
    ax.plot([-mx, mx, mx, -mx, -mx], [-my, -my, my, my, -my], color="#1f7a3a", lw=1.3, ls="--", zorder=3)

    for r in ROOMS:
        (wx, wy), (nx, ny) = wall_point_and_normal(r["wall"], r["t"])
        yaw = math.degrees(math.atan2(ny, nx)) - 90
        cx, cy = wx + nx * (WALL_T + r["depth"] / 2), wy + ny * (WALL_T + r["depth"] / 2)
        rect = Rectangle((-r["width"] / 2, -r["depth"] / 2), r["width"], r["depth"],
                         fc=[*r["color"]], ec="#3a8f4f" if r["object"] is None else "#c48a10", lw=1.8)
        t = plt.matplotlib.transforms.Affine2D().rotate_deg(yaw).translate(cx, cy) + ax.transData
        rect.set_transform(t)
        ax.add_patch(rect)
        door_x, door_y = wx + nx * 0.15, wy + ny * 0.15
        ax.plot(door_x, door_y, "s", color="#8a2f5a", ms=6, zorder=4)
        label_x, label_y = wx + nx * (WALL_T + r["depth"] + 0.35), wy + ny * (WALL_T + r["depth"] + 0.35)
        obj_label = r["object"]["kind"].replace("_", " ") if r["object"] else "EMPTY - reference"
        ax.text(label_x, label_y, f"{r['name']}\n{obj_label}", ha="center", va="center", fontsize=8.6, fontweight="bold",
                color="#1f3a5f" if r["object"] is None else "#7a5200")

    sx, sy = gt["robot_spawn"]["x"], gt["robot_spawn"]["y"]
    ax.plot(sx, sy, "o", color="#1f7a3a", ms=9, zorder=5)
    ax.annotate("robot start\n(facing +x)", (sx, sy), (sx + 0.6, sy - 0.7), fontsize=8.5, color="#1f7a3a",
                arrowprops=dict(arrowstyle="->", color="#1f7a3a"))

    ax.set_title("UrbanLens rectangular loop (top view) -- placeholder scale, not final", fontsize=12, fontweight="bold")
    ax.set_xlim(-ux - 2.2, ux + 2.2); ax.set_ylim(-uy - 2.2, uy + 2.6)
    ax.set_aspect("equal"); ax.axis("off")
    fig.savefig(os.path.join(HERE, "loop_layout_preview.png"), bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    with open(os.path.join(HERE, "urbanlens_loop.world"), "w") as f:
        f.write(build_world())
    gt = build_ground_truth()
    with open(os.path.join(HERE, "loop_ground_truth.json"), "w") as f:
        json.dump(gt, f, indent=2)
    print("Wrote urbanlens_loop.world and loop_ground_truth.json")
    for name, s in gt["segments"].items():
        print(f"  {name}: {s['wall']} wall, room {s['room_width_m']}x{s['room_depth_m']} m, "
              f"{'REFERENCE (empty)' if s['is_reference'] else 'object: ' + s['object']}")
    try:
        draw_preview(gt)
        print("Wrote loop_layout_preview.png")
    except Exception as e:
        print("(preview skipped:", e, ")")

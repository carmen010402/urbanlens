"""
make_corridor_world.py

Builds the UrbanLens TEST CORRIDOR for Gazebo, plus the "manually verified
layout" (ground truth) that goes with it. Both files are generated from the
SAME list of obstacles below, so the world and the answer key can never
disagree.

  corridor: 1.50 m wide, 10 m long, 5 segments of 2 m (S1..S5), start pad behind x = 0
  robot spawns at x = 0, y = 0, facing +x  (so the Mission Manager's distance
  counter and the corridor's segments line up when you press START at the start)

HOW TO CHANGE THE TEST: edit OBSTACLES, run:  python3 make_corridor_world.py
It writes  urbanlens_corridor.world  and  corridor_ground_truth.json  (and a
preview picture  corridor_layout.png  if matplotlib is installed).

Why the "true clearance" is exact: the Mission Manager measures clearance as
(LiDAR range to the left) + (LiDAR range to the right). With straight walls and
the robot pointing down the corridor, that sum equals the distance between the
two facing surfaces, wherever the robot stands sideways. So it equals
corridor width minus the obstacle's width.
"""
import json
import os

CORRIDOR_WIDTH = 1.50
SEGMENT_LENGTH = 2.0
N_SEGMENTS = 5
WALL_T, WALL_H = 0.10, 1.0
BACK_PAD = 0.5                      # free space behind x = 0 where the robot spawns
L = SEGMENT_LENGTH * N_SEGMENTS     # corridor length (10 m)

# One staged obstruction per segment at most. side = wall it stands against.
OBSTACLES = [
    dict(name="box_S2", segment=2, side="left",  width=0.30, length=1.0, height=0.5, x_center=3.2, color=(0.95, 0.55, 0.10)),
    dict(name="box_S4", segment=4, side="right", width=0.65, length=1.0, height=0.5, x_center=7.2, color=(0.85, 0.25, 0.15)),
    dict(name="box_S5", segment=5, side="left",  width=0.85, length=1.0, height=0.5, x_center=9.0, color=(0.75, 0.15, 0.55)),
]

HERE = os.path.dirname(os.path.abspath(__file__))


def box_model(name, x, y, z, sx, sy, sz, color, collision=True):
    r, g, b = color
    col = (f"""
      <collision name="collision"><geometry><box><size>{sx} {sy} {sz}</size></box></geometry></collision>"""
           if collision else "")
    return f"""
    <model name="{name}">
      <static>true</static>
      <pose>{x:.4f} {y:.4f} {z:.4f} 0 0 0</pose>
      <link name="link">{col}
        <visual name="visual">
          <geometry><box><size>{sx} {sy} {sz}</size></box></geometry>
          <material><ambient>{r} {g} {b} 1</ambient><diffuse>{r} {g} {b} 1</diffuse></material>
        </visual>
      </link>
    </model>"""


def build_world():
    parts = []
    half = CORRIDOR_WIDTH / 2
    # side walls (long), back wall, end wall
    wall_len = L + BACK_PAD + 2 * WALL_T
    wall_cx = (-BACK_PAD - WALL_T + L + WALL_T) / 2
    grey = (0.78, 0.78, 0.82)
    for side, sgn in (("left", 1), ("right", -1)):
        parts.append(box_model(f"wall_{side}", wall_cx, sgn * (half + WALL_T / 2), WALL_H / 2,
                               round(wall_len, 3), WALL_T, WALL_H, grey))
    span = CORRIDOR_WIDTH + 2 * WALL_T
    parts.append(box_model("wall_back", -BACK_PAD - WALL_T / 2, 0, WALL_H / 2, WALL_T, span, WALL_H, grey))
    parts.append(box_model("wall_end", L + WALL_T / 2, 0, WALL_H / 2, WALL_T, span, WALL_H, grey))

    # floor colouring per segment + boundary lines (visual only: LiDAR ignores them)
    tile_cols = [(0.72, 0.85, 0.95), (0.80, 0.92, 0.78)]
    parts.append(box_model("floor_start_pad", -BACK_PAD / 2, 0, 0.002, BACK_PAD, CORRIDOR_WIDTH, 0.004, (0.6, 0.6, 0.6), False))
    for i in range(N_SEGMENTS):
        parts.append(box_model(f"floor_S{i+1}", i * SEGMENT_LENGTH + SEGMENT_LENGTH / 2, 0, 0.002,
                               SEGMENT_LENGTH, CORRIDOR_WIDTH, 0.004, tile_cols[i % 2], False))
    for k in range(N_SEGMENTS + 1):
        parts.append(box_model(f"line_{k}", k * SEGMENT_LENGTH, 0, 0.004, 0.05, CORRIDOR_WIDTH, 0.006, (0.85, 0.1, 0.1), False))

    # staged obstacles
    for o in OBSTACLES:
        y = (half - o["width"] / 2) * (1 if o["side"] == "left" else -1)
        parts.append(box_model(o["name"], o["x_center"], y, o["height"] / 2,
                               o["length"], o["width"], o["height"], o["color"]))

    return f"""<?xml version="1.0" ?>
<sdf version="1.6">
  <world name="urbanlens_corridor">
    <scene><ambient>0.55 0.55 0.55 1</ambient><background>0.75 0.82 0.9 1</background><shadows>true</shadows></scene>
    <light name="sun" type="directional">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.85 0.85 0.85 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <attenuation><range>1000</range><constant>0.9</constant><linear>0.01</linear><quadratic>0.001</quadratic></attenuation>
      <direction>-0.4 0.2 -0.9</direction>
    </light>
    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry><plane><normal>0 0 1</normal><size>60 60</size></plane></geometry>
          <surface><friction><ode><mu>100</mu><mu2>50</mu2></ode></friction></surface>
        </collision>
        <visual name="visual">
          <cast_shadows>false</cast_shadows>
          <geometry><plane><normal>0 0 1</normal><size>60 60</size></plane></geometry>
          <material><ambient>0.4 0.4 0.4 1</ambient><diffuse>0.4 0.4 0.4 1</diffuse></material>
        </visual>
      </link>
    </model>
{''.join(parts)}
  </world>
</sdf>
"""


def build_ground_truth():
    segs = {}
    for i in range(1, N_SEGMENTS + 1):
        obs = [o for o in OBSTACLES if o["segment"] == i]
        assert len(obs) <= 1, "this generator supports at most one obstacle per segment"
        o = obs[0] if obs else None
        true_clear = round(CORRIDOR_WIDTH - (o["width"] if o else 0.0), 3)
        segs[f"S{i}"] = {
            "x_start": (i - 1) * SEGMENT_LENGTH, "x_end": i * SEGMENT_LENGTH,
            "reference_clearance_m": CORRIDOR_WIDTH,
            "true_clearance_m": true_clear,
            "staged_obstacle": (f"{o['name']}: {o['width']} m wide x {o['length']} m long, on the {o['side']} wall, "
                                f"x = {o['x_center'] - o['length']/2:.1f} to {o['x_center'] + o['length']/2:.1f} m") if o else None,
        }
    return {"description": "Manually verified layout of the simulated UrbanLens test corridor",
            "corridor": {"width_m": CORRIDOR_WIDTH, "length_m": L, "segment_length_m": SEGMENT_LENGTH,
                         "robot_spawn": {"x": 0.0, "y": 0.0, "facing": "+x"}},
            "segments": segs}


def draw_preview(gt):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig, ax = plt.subplots(figsize=(12, 3.6), dpi=150)
    half = CORRIDOR_WIDTH / 2
    cols = ["#dbeaf7", "#e6f2e2"]
    for i in range(N_SEGMENTS):
        ax.add_patch(Rectangle((i * SEGMENT_LENGTH, -half), SEGMENT_LENGTH, CORRIDOR_WIDTH, fc=cols[i % 2], ec="#c0392b", lw=1.2))
        s = gt["segments"][f"S{i+1}"]
        ax.text(i * SEGMENT_LENGTH + SEGMENT_LENGTH / 2, half + 0.28, f"S{i+1}", ha="center", fontsize=12, fontweight="bold")
        ax.text(i * SEGMENT_LENGTH + SEGMENT_LENGTH / 2, -half - 0.42, f"true clearance {s['true_clearance_m']:.2f} m",
                ha="center", fontsize=9, color="#1f3a5f")
    ax.add_patch(Rectangle((-BACK_PAD, -half), BACK_PAD, CORRIDOR_WIDTH, fc="#dddddd", ec="none"))
    for o in OBSTACLES:
        y0 = (half - o["width"]) if o["side"] == "left" else -half
        ax.add_patch(Rectangle((o["x_center"] - o["length"] / 2, y0), o["length"], o["width"], fc=[*o["color"]], ec="black"))
        ax.text(o["x_center"], y0 + o["width"] / 2, f"{o['width']:.2f} m", ha="center", va="center", color="white", fontsize=9, fontweight="bold")
    for sgn in (1, -1):
        ax.add_patch(Rectangle((-BACK_PAD - WALL_T, sgn * half if sgn > 0 else -half - WALL_T), L + BACK_PAD + 2 * WALL_T, WALL_T, fc="#555", ec="none"))
    ax.plot([0], [0], "o", color="#1f7a3a", ms=9); ax.annotate("robot start", (0, 0), (-0.45, 0.55), fontsize=9, color="#1f7a3a",
                                                              arrowprops=dict(arrowstyle="->", color="#1f7a3a"))
    ax.set_xlim(-0.9, L + 0.4); ax.set_ylim(-1.6, 1.5); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("UrbanLens test corridor (top view): reference width 1.50 m", fontsize=12, fontweight="bold")
    fig.savefig(os.path.join(HERE, "corridor_layout.png"), bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    with open(os.path.join(HERE, "urbanlens_corridor.world"), "w") as f:
        f.write(build_world())
    gt = build_ground_truth()
    with open(os.path.join(HERE, "corridor_ground_truth.json"), "w") as f:
        json.dump(gt, f, indent=2)
    print("Wrote urbanlens_corridor.world and corridor_ground_truth.json")
    for k, v in gt["segments"].items():
        print(f"  {k}: true clearance {v['true_clearance_m']:.2f} m   {v['staged_obstacle'] or '(clear)'}")
    try:
        draw_preview(gt); print("Wrote corridor_layout.png")
    except Exception as e:
        print("(preview picture skipped:", e, ")")

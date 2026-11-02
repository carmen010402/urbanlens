"""
segment_manager.py

This is the missing piece between your detection code (detect.py) and your
teammate's dashboard. It:
  1. Stores each of your rover's route segments (S1, S2, S3...)
  2. Records findings (reference clearance vs. actual clearance, severity,
     LiDAR reading, robot position, camera confidence, timestamp)
  3. Tracks coverage: surveyed / not surveyed / has a finding
  4. Generates the exact MISSION REPORT text your team hand-wrote on paper

Everything here matches the fields in your own hand-drawn evidence panel and
mission report sketches, field for field:
  segment, reference, actual, reduction, severity, LiDAR distance,
  camera confidence, robot x/y/orientation, timestamp, object type.

HOW YOUR TEAMMATE'S DASHBOARD USES THIS:
Their frontend doesn't need to calculate anything -- it just calls the
functions below (directly, or through a small API wrapper around them) and
displays whatever comes back. detect.py's job is only to say "I saw a chair,
here's roughly where" -- record_finding() below does the rest: computing
reduction, severity, and updating coverage.
"""

import sqlite3
import math
import os
from datetime import datetime
from dataclasses import dataclass
from typing import Optional

# FIX: this used to be just "urbanlens.db", which meant the actual file
# location depended on whatever folder you happened to run a command from --
# if you ran the node from one folder and the report from another, you'd
# silently create/read two different empty databases and always see zeros.
# Anchoring it to this file's own folder means it's ALWAYS the same file,
# no matter which directory your terminal is sitting in when you run something.
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "urbanlens.db")


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def init_db(db_path: str = DB_PATH, reset: bool = False):
    """Creates the database and tables. Call this once when the mission starts."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    if reset:
        c.execute("DROP TABLE IF EXISTS segments")
        c.execute("DROP TABLE IF EXISTS findings")
        c.execute("DROP TABLE IF EXISTS mission")

    c.execute("""
        CREATE TABLE IF NOT EXISTS segments (
            segment_id TEXT PRIMARY KEY,
            reference_clearance_m REAL NOT NULL,
            surveyed INTEGER DEFAULT 0,
            last_actual_clearance_m REAL,
            last_finding_type TEXT,
            last_object_type TEXT,
            last_severity INTEGER,
            last_updated TEXT
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS findings (
            finding_id INTEGER PRIMARY KEY AUTOINCREMENT,
            segment_id TEXT NOT NULL,
            finding_type TEXT NOT NULL,
            object_type TEXT,
            reference_clearance_m REAL NOT NULL,
            actual_clearance_m REAL NOT NULL,
            reduction_m REAL NOT NULL,
            severity INTEGER NOT NULL,
            lidar_distance_m REAL,
            camera_confidence REAL,
            robot_x REAL,
            robot_y REAL,
            robot_orientation_deg REAL,
            timestamp TEXT NOT NULL,
            image_path TEXT,
            FOREIGN KEY (segment_id) REFERENCES segments(segment_id)
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS mission (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            status TEXT DEFAULT 'NOT_STARTED',
            started_at TEXT,
            stopped_at TEXT,
            distance_m REAL DEFAULT 0.0,
            unique_people_count INTEGER DEFAULT 0
        )
    """)
    c.execute("INSERT OR IGNORE INTO mission (id) VALUES (1)")

    conn.commit()
    conn.close()


def define_segments(segment_reference_map: dict, db_path: str = DB_PATH):
    """
    Call once, at the start, to define your route's segments and their
    REFERENCE (expected/design) clearance -- e.g. the corridor's designed
    width when nothing is blocking it.

    Example: define_segments({"S1": 1.50, "S2": 1.50, "S3": 1.50, "S4": 1.50, "S5": 1.50})
    """
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    for seg_id, ref_clearance in segment_reference_map.items():
        c.execute("""
            INSERT OR IGNORE INTO segments (segment_id, reference_clearance_m)
            VALUES (?, ?)
        """, (seg_id, ref_clearance))
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Severity calculation
# ---------------------------------------------------------------------------

def compute_severity(reference_m: float, actual_m: float) -> int:
    """
    Severity 1-5 based on how much the actual clearance shrank compared to
    the reference. Tune these thresholds to match what your team considers
    "minor" vs "severe" -- these defaults are a reasonable starting point:

    reduction %      severity
    0%  - 10%    ->  1
    10% - 25%    ->  2
    25% - 45%    ->  3
    45% - 65%    ->  4
    65% - 100%   ->  5  (path nearly or fully blocked)
    """
    if reference_m <= 0:
        return 1
    reduction_pct = max(0.0, (reference_m - actual_m) / reference_m)

    if reduction_pct < 0.10:
        return 1
    elif reduction_pct < 0.25:
        return 2
    elif reduction_pct < 0.45:
        return 3
    elif reduction_pct < 0.65:
        return 4
    else:
        return 5


# ---------------------------------------------------------------------------
# Recording findings -- this is what detect.py's output feeds into
# ---------------------------------------------------------------------------

def record_finding(segment_id: str, object_type: str, actual_clearance_m: float,
                    lidar_distance_m: Optional[float] = None,
                    camera_confidence: Optional[float] = None,
                    robot_x: Optional[float] = None, robot_y: Optional[float] = None,
                    robot_orientation_deg: Optional[float] = None,
                    image_path: Optional[str] = None,
                    db_path: str = DB_PATH) -> dict:
    """
    Call this every time detect.py identifies an obstruction/narrow path/etc.
    in a given segment. Computes reduction + severity automatically and
    updates that segment's status.

    If actual_clearance_m == reference_clearance_m (nothing blocking), this
    marks the segment as surveyed with NO finding -- matching your S1 example
    ("Surveyed: YES, Finding: None").
    """
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    c.execute("SELECT reference_clearance_m FROM segments WHERE segment_id = ?", (segment_id,))
    row = c.fetchone()
    if row is None:
        conn.close()
        raise ValueError(f"Segment '{segment_id}' was never defined. Call define_segments() first.")
    reference_m = row[0]

    reduction_m = round(reference_m - actual_clearance_m, 3)
    timestamp = datetime.now().strftime("%H:%M:%S")

    has_finding = reduction_m > 0.02  # a couple cm of noise tolerance

    if has_finding:
        severity = compute_severity(reference_m, actual_clearance_m)
        finding_type = "Narrow path" if actual_clearance_m < reference_m * 0.5 else "Obstruction"

        c.execute("""
            INSERT INTO findings (segment_id, finding_type, object_type, reference_clearance_m,
                                   actual_clearance_m, reduction_m, severity, lidar_distance_m,
                                   camera_confidence, robot_x, robot_y, robot_orientation_deg,
                                   timestamp, image_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (segment_id, finding_type, object_type, reference_m, actual_clearance_m,
              reduction_m, severity, lidar_distance_m, camera_confidence,
              robot_x, robot_y, robot_orientation_deg, timestamp, image_path))

        c.execute("""
            UPDATE segments SET surveyed = 1, last_actual_clearance_m = ?,
                   last_finding_type = ?, last_object_type = ?, last_severity = ?,
                   last_updated = ?
            WHERE segment_id = ?
        """, (actual_clearance_m, finding_type, object_type, severity, timestamp, segment_id))
    else:
        severity = None
        finding_type = None
        c.execute("""
            UPDATE segments SET surveyed = 1, last_actual_clearance_m = ?,
                   last_finding_type = NULL, last_object_type = NULL,
                   last_severity = NULL, last_updated = ?
            WHERE segment_id = ?
        """, (actual_clearance_m, timestamp, segment_id))

    conn.commit()
    conn.close()

    return {
        "segment_id": segment_id, "reference_m": reference_m, "actual_m": actual_clearance_m,
        "reduction_m": reduction_m, "severity": severity, "finding_type": finding_type,
        "has_finding": has_finding,
    }


# ---------------------------------------------------------------------------
# Reading back data -- this is what the dashboard calls
# ---------------------------------------------------------------------------

def update_mission_progress(distance_m: Optional[float] = None,
                             unique_people_count: Optional[int] = None,
                             status: Optional[str] = None,
                             db_path: str = DB_PATH):
    """
    Call this periodically (e.g. from mission_manager_node's timer) to keep
    the mission table's totals current. Only updates the fields you pass in
    -- leave the others as None to leave them unchanged.
    """
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    if distance_m is not None:
        c.execute("UPDATE mission SET distance_m = ? WHERE id = 1", (distance_m,))
    if unique_people_count is not None:
        c.execute("UPDATE mission SET unique_people_count = ? WHERE id = 1", (unique_people_count,))
    if status is not None:
        c.execute("UPDATE mission SET status = ? WHERE id = 1", (status,))
    conn.commit()
    conn.close()


def get_coverage(db_path: str = DB_PATH) -> list[dict]:
    """
    Matches your COVERAGE row sketch: for each segment, is it surveyed, and
    does it have a finding? ('checkmark' / 'warning triangle' / 'circle')
    """
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT segment_id, surveyed, last_finding_type FROM segments ORDER BY segment_id")
    rows = c.fetchall()
    conn.close()

    result = []
    for seg_id, surveyed, finding_type in rows:
        if not surveyed:
            status = "not_surveyed"       # your "○"
        elif finding_type:
            status = "finding_detected"   # your "⚠"
        else:
            status = "surveyed_clear"     # your "✓"
        result.append({"segment_id": seg_id, "status": status})
    return result


def get_findings_table(db_path: str = DB_PATH) -> list[dict]:
    """Matches your FINDINGS table sketch exactly: Seg | Finding | Reference | Actual | Severity."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""
        SELECT segment_id, finding_type, reference_clearance_m, actual_clearance_m, severity, timestamp
        FROM findings ORDER BY finding_id
    """)
    rows = c.fetchall()
    conn.close()
    return [
        {"segment": r[0], "finding": r[1], "reference_m": r[2], "actual_m": r[3],
         "severity": r[4], "timestamp": r[5]}
        for r in rows
    ]


def get_segment_evidence(segment_id: str, db_path: str = DB_PATH) -> Optional[dict]:
    """Matches your EVIDENCE panel sketch: everything about that segment's most recent finding."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""
        SELECT * FROM findings WHERE segment_id = ? ORDER BY finding_id DESC LIMIT 1
    """, (segment_id,))
    row = c.fetchone()
    if row is None:
        conn.close()
        return None
    cols = [d[0] for d in c.description]
    conn.close()
    return dict(zip(cols, row))


def generate_mission_report(db_path: str = DB_PATH) -> dict:
    """
    Matches your hand-written MISSION REPORT exactly:
    coverage %, distance, segments surveyed, unique people, findings count,
    and the single largest clearance reduction (with its full evidence).
    """
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    c.execute("SELECT COUNT(*), SUM(surveyed) FROM segments")
    total_segments, surveyed_segments = c.fetchone()
    surveyed_segments = surveyed_segments or 0
    coverage_pct = round(100 * surveyed_segments / total_segments) if total_segments else 0

    c.execute("SELECT distance_m, unique_people_count FROM mission WHERE id = 1")
    distance_m, unique_people = c.fetchone()

    c.execute("SELECT COUNT(*) FROM findings")
    findings_count = c.fetchone()[0]

    c.execute("""
        SELECT segment_id, finding_type, object_type, reference_clearance_m, actual_clearance_m,
               reduction_m, severity, image_path, lidar_distance_m, robot_x, robot_y, timestamp
        FROM findings ORDER BY reduction_m DESC LIMIT 1
    """)
    worst = c.fetchone()
    conn.close()

    report = {
        "mission_status": "Completed",
        "coverage_pct": coverage_pct,
        "distance_m": distance_m,
        "segments_surveyed": f"{surveyed_segments}/{total_segments}",
        "unique_people": unique_people,
        "findings_count": findings_count,
    }

    if worst:
        report["largest_reduction"] = {
            "segment": worst[0], "finding_type": worst[1], "object_type": worst[2],
            "reference_m": worst[3], "actual_m": worst[4], "reduction_m": worst[5],
            "severity": worst[6], "image_path": worst[7], "lidar_distance_m": worst[8],
            "robot_x": worst[9], "robot_y": worst[10], "timestamp": worst[11],
        }
    return report


def print_mission_report(db_path: str = DB_PATH):
    """Pretty-prints the report the same way your hand-written sketch reads."""
    r = generate_mission_report(db_path)
    print("MISSION REPORT\n")
    print(f"Mission status: {r['mission_status']}")
    print(f"Coverage: {r['coverage_pct']}%")
    print(f"Distance: {r['distance_m']}m")
    print(f"Segments surveyed: {r['segments_surveyed']}")
    print(f"Unique people: {r['unique_people']}")
    print(f"Findings: {r['findings_count']}")
    if "largest_reduction" in r:
        lr = r["largest_reduction"]
        print(f"\nLargest clearance reduction: {lr['segment']} - {lr['actual_m']}m")
        print(f"\nFinding: {lr['finding_type']}")
        print(f"Object: {lr['object_type']}")
        print(f"Reference: {lr['reference_m']}m")
        print(f"Measured: {lr['actual_m']}m")
        print(f"Reduction: {lr['reduction_m']}m")
        print(f"Severity: {lr['severity']}/5")
        print("\nEvidence:")
        print(f"- Camera image [{lr['segment']}]")
        print(f"- LiDAR observation ({lr['lidar_distance_m']}m)")
        print(f"- Robot position (x:{lr['robot_x']}, y:{lr['robot_y']})")
        print(f"- Timestamp {lr['timestamp']}")


# ---------------------------------------------------------------------------
# Demo: reproduces YOUR hand-written example numbers exactly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    init_db(reset=True)
    define_segments({"S1": 1.50, "S2": 1.50, "S3": 1.50, "S4": 1.50, "S5": 1.50})

    # S1: surveyed, nothing found (matches your handwritten example exactly)
    record_finding("S1", object_type=None, actual_clearance_m=1.50,
                    lidar_distance_m=None, robot_x=0.0, robot_y=0.0)

    # S2: obstruction, mild
    record_finding("S2", object_type="Obstruction", actual_clearance_m=1.20,
                    lidar_distance_m=1.10, camera_confidence=0.88, robot_x=1.0, robot_y=0.2)

    # S3: clear
    record_finding("S3", object_type=None, actual_clearance_m=1.50, robot_x=2.0, robot_y=0.3)

    # S4: chair obstruction (matches your CURRENT SEGMENT sketch exactly)
    record_finding("S4", object_type="Chair", actual_clearance_m=0.85,
                    lidar_distance_m=0.72, camera_confidence=0.91,
                    robot_x=2.41, robot_y=1.18, robot_orientation_deg=90,
                    image_path="S4_evidence.jpg")

    # S5: narrow path, worst case (matches your report's "largest reduction" example)
    record_finding("S5", object_type="Narrow path", actual_clearance_m=0.65,
                    lidar_distance_m=0.60, robot_x=3.5, robot_y=1.0,
                    image_path="S5_evidence.jpg")

    print("=== COVERAGE ===")
    for seg in get_coverage():
        print(seg)

    print("\n=== FINDINGS TABLE ===")
    for f in get_findings_table():
        print(f)

    print("\n" + "=" * 50)
    print_mission_report()
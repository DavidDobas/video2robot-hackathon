#!/usr/bin/env python3
"""
Convert robot motion PKL files to LAFAN dataset CSV format

Usage:
    python scripts/convert_to_lafan_csv.py --project data/video_010
    python scripts/convert_to_lafan_csv.py --pkl data/video_010/robot_motion_track_1.pkl
"""

import argparse
import pickle
import csv
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from video2robot.config import DATA_DIR

# LAFAN joint order (columns 7-35 in CSV)
LAFAN_JOINT_ORDER = [
    "left_hip_pitch",
    "left_hip_roll",
    "left_hip_yaw",
    "left_knee",
    "left_ankle_pitch",
    "left_ankle_roll",
    "right_hip_pitch",
    "right_hip_roll",
    "right_hip_yaw",
    "right_knee",
    "right_ankle_pitch",
    "right_ankle_roll",
    "waist_yaw",
    "waist_roll",
    "waist_pitch",
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_shoulder_yaw",
    "left_elbow",
    "left_wrist_roll",
    "left_wrist_pitch",
    "left_wrist_yaw",
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow",
    "right_wrist_roll",
    "right_wrist_pitch",
    "right_wrist_yaw",
]

# GMR unitree_g1 joint order (from dof_pos)
GMR_JOINT_ORDER = [
    "left_hip_pitch",      # 0
    "left_hip_roll",       # 1
    "left_hip_yaw",        # 2
    "left_knee",           # 3
    "left_ankle_pitch",    # 4
    "left_ankle_roll",     # 5
    "right_hip_pitch",     # 6
    "right_hip_roll",      # 7
    "right_hip_yaw",       # 8
    "right_knee",          # 9
    "right_ankle_pitch",   # 10
    "right_ankle_roll",    # 11
    "waist_yaw",           # 12
    "waist_roll",          # 13
    "waist_pitch",         # 14
    "left_shoulder_pitch", # 15
    "left_shoulder_roll",  # 16
    "left_shoulder_yaw",   # 17
    "left_elbow",          # 18
    "left_wrist_roll",     # 19
    "left_wrist_pitch",    # 20
    "left_wrist_yaw",      # 21 (if 29dof)
    "right_shoulder_pitch",# 22
    "right_shoulder_roll", # 23
    "right_shoulder_yaw",  # 24
    "right_elbow",         # 25
    "right_wrist_roll",    # 26
    "right_wrist_pitch",   # 27
    "right_wrist_yaw",     # 28
]


def create_joint_mapping():
    """Create mapping from GMR joint order to LAFAN joint order"""
    mapping = []
    for lafan_joint in LAFAN_JOINT_ORDER:
        if lafan_joint in GMR_JOINT_ORDER:
            mapping.append(GMR_JOINT_ORDER.index(lafan_joint))
        else:
            raise ValueError(f"Joint {lafan_joint} not found in GMR joint order")
    return mapping


def convert_pkl_to_csv(pkl_path: Path, output_path: Path = None):
    """Convert robot motion PKL to LAFAN CSV format."""
    with open(pkl_path, "rb") as f:
        robot_motion = pickle.load(f)

    root_pos = np.array(robot_motion["root_pos"])  # (N, 3)
    root_rot = np.array(robot_motion["root_rot"])  # (N, 4) xyzw
    dof_pos = np.array(robot_motion["dof_pos"])    # (N, 29)

    num_frames = root_pos.shape[0]

    if dof_pos.shape[1] != 29:
        raise ValueError(f"Expected 29 joints, got {dof_pos.shape[1]}. Only unitree_g1 29dof is supported.")

    joint_mapping = create_joint_mapping()
    reordered_dof = dof_pos[:, joint_mapping]

    if output_path is None:
        output_path = pkl_path.with_suffix(".csv")

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        for i in range(num_frames):
            row = []
            row.extend(root_pos[i].tolist())   # x, y, z
            row.extend(root_rot[i].tolist())   # qx, qy, qz, qw
            row.extend(reordered_dof[i].tolist())
            writer.writerow(row)

    print(f"Converted {num_frames} frames to CSV: {output_path}")
    print(f"  FPS: {robot_motion.get('fps', 'N/A')}")
    print(f"  Robot: {robot_motion.get('robot_type', 'N/A')}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert robot motion PKL to LAFAN CSV format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/convert_to_lafan_csv.py --project data/video_010
  python scripts/convert_to_lafan_csv.py --project data/video_010 --all-tracks
  python scripts/convert_to_lafan_csv.py --pkl data/video_010/robot_motion_track_1.pkl
        """
    )
    parser.add_argument("--project", "-p", help="Project folder path")
    parser.add_argument("--pkl", help="PKL file path (alternative to --project)")
    parser.add_argument("--output", "-o", help="Output CSV path (default: same as input with .csv extension)")
    parser.add_argument("--all-tracks", action="store_true", help="Convert all robot_motion_track_*.pkl files")

    args = parser.parse_args()

    if not args.project and not args.pkl:
        parser.error("Provide --project or --pkl")

    if args.project:
        project_dir = Path(args.project)
        if not project_dir.exists():
            parser.error(f"Project not found: {project_dir}")

        if args.all_tracks:
            pkl_files = sorted(project_dir.glob("robot_motion_track_*.pkl"))
            pkl_files = [p for p in pkl_files if "twist" not in p.name]
        else:
            default_pkl = project_dir / "robot_motion.pkl"
            if default_pkl.exists():
                pkl_files = [default_pkl]
            else:
                track1_pkl = project_dir / "robot_motion_track_1.pkl"
                if track1_pkl.exists():
                    pkl_files = [track1_pkl]
                else:
                    parser.error(f"No robot motion PKL found in {project_dir}")

        for pkl_file in pkl_files:
            convert_pkl_to_csv(pkl_file, args.output if args.output else None)
    else:
        pkl_path = Path(args.pkl)
        if not pkl_path.exists():
            parser.error(f"PKL file not found: {pkl_path}")
        convert_pkl_to_csv(pkl_path, args.output)


if __name__ == "__main__":
    main()

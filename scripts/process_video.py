#!/usr/bin/env python3
"""
Run pose extraction + robot conversion + LAFAN CSV export in one command.

Usage:
    python scripts/process_video.py --project data/video_001

    # With optional flags passed through to the individual steps
    python scripts/process_video.py --project data/video_001 --static-camera
    python scripts/process_video.py --project data/video_001 --no-twist
"""

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent


def run(script: str, extra_args: list[str]) -> None:
    cmd = [sys.executable, str(SCRIPTS_DIR / script)] + extra_args
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\nERROR: {script} failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description="Extract pose, convert to robot motion, and export LAFAN CSV")
    parser.add_argument("--project", required=True, help="Project directory (e.g. data/video_001)")
    parser.add_argument("--static-camera", action="store_true", help="Assume static camera (passed to extract_pose)")
    parser.add_argument("--no-twist", action="store_true", help="Skip TWIST compatibility (passed to convert_to_robot)")
    args = parser.parse_args()

    project_args = ["--project", args.project]

    run("extract_pose.py", project_args + (["--static-camera"] if args.static_camera else []))
    run("convert_to_robot.py", project_args + (["--no-twist"] if args.no_twist else []))
    run("convert_to_lafan_csv.py", project_args)

    print(f"\nDone! All steps completed for project: {args.project}")


if __name__ == "__main__":
    main()

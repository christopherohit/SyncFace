#!/usr/bin/env python3
"""
Auto-run script for SyncFace - Process all videos in a folder with process.py

Usage:
    python run_process_all_videos.py <path_to_video_or_folder>
    python run_process_all_videos.py data/pretrain/Jae-in/
    python run_process_all_videos.py data/pretrain/Jae-in/Jae-in.mp4
    python run_process_all_videos.py .  # Process all videos in current directory

Options:
    --task TASK       Task number to run (-1 for all tasks, default: -1)
    --asr ASR         Audio feature extractor (ave, hubert, deepspeech, default: ave)
    --parallel PAR    Number of parallel processes (default: 1, sequential)
    --dry-run         Show what would be done without actually running
    --skip SKIP       Skip specific tasks (e.g., --skip 3,4 for skip tasks 3 and 4)
"""

import os
import sys
import glob
import argparse
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple
import time

# Video extensions to search for
VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v']


def find_videos(path: str) -> List[str]:
    """
    Find all video files in the given path (file or directory).
    
    Args:
        path: Path to a video file or directory containing videos
        
    Returns:
        List of absolute paths to video files
    """
    videos = []
    path = os.path.abspath(path)
    
    if os.path.isfile(path):
        if any(path.lower().endswith(ext) for ext in VIDEO_EXTENSIONS):
            videos.append(path)
    elif os.path.isdir(path):
        for ext in VIDEO_EXTENSIONS:
            videos.extend(glob.glob(os.path.join(path, '**', '*' + ext), recursive=True))
            videos.extend(glob.glob(os.path.join(path, '*' + ext)))
    
    return sorted(videos)


def run_process_for_video(video_path: str, task: int = -1, asr: str = 'ave') -> Tuple[str, bool, str]:
    """
    Run process.py for a single video file.
    
    Args:
        video_path: Path to the video file
        task: Task number to run (-1 for all)
        asr: Audio feature extractor type
        
    Returns:
        Tuple of (video_path, success, message)
    """
    try:
        # Change to SyncFace root directory (parent of tools/)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        syncface_dir = os.path.dirname(script_dir)  # Go up one level from tools/
        os.chdir(syncface_dir)
        
        # Build command
        cmd = [
            sys.executable,
            'data_utils_enhancement/process.py',
            video_path,
            '--task', str(task),
            '--asr', asr
        ]
        
        print(f"\n{'='*60}")
        print(f"Processing: {video_path}")
        print(f"Command: {' '.join(cmd)}")
        print(f"{'='*60}")
        
        # Run the process
        result = subprocess.run(
            cmd,
            capture_output=False,
            text=True
        )
        
        success = result.returncode == 0
        
        if success:
            return video_path, True, "Success"
        else:
            return video_path, False, f"Failed with return code {result.returncode}"
            
    except Exception as e:
        return video_path, False, f"Error: {str(e)}"


def process_video_with_retry(video_path: str, task: int = -1, asr: str = 'ave', 
                             max_retries: int = 2, retry_delay: int = 5) -> Tuple[str, bool, str]:
    """
    Process a video with retry logic.
    """
    for attempt in range(max_retries):
        video_path, success, message = run_process_for_video(video_path, task, asr)
        
        if success:
            return video_path, True, message
        
        if attempt < max_retries - 1:
            print(f"\n⚠️  Retrying in {retry_delay} seconds... (Attempt {attempt + 2}/{max_retries})")
            time.sleep(retry_delay)
    
    return video_path, False, message


def main():
    parser = argparse.ArgumentParser(
        description='Auto-run SyncFace process.py for all videos in a folder',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process all videos in a directory
    python run_process_all_videos.py data/pretrain/
    
    # Process a single video
    python run_process_all_videos.py data/pretrain/Jae-in/Jae-in.mp4
    
    # Process with specific task only (e.g., extract images only)
    python run_process_all_videos.py data/pretrain/ --task 2
    
    # Run multiple tasks (e.g., tasks 2 and 3)
    python run_process_all_videos.py data/pretrain/ --task 2 --skip 1,3,4
    
    # Dry run - show what would be processed
    python run_process_all_videos.py data/pretrain/ --dry-run
    
    # Process with hubert audio features
    python run_process_all_videos.py data/pretrain/ --asr hubert
        """
    )
    
    parser.add_argument('path', type=str, nargs='?', default='.',
                        help='Path to video file or directory (default: current directory)')
    parser.add_argument('--task', type=int, default=-1,
                        help='Task number (-1 for all, 1-10 for specific task, default: -1)')
    parser.add_argument('--asr', type=str, default='ave',
                        choices=['ave', 'hubert', 'deepspeech', 'wav2vec'],
                        help='Audio feature extractor (default: ave)')
    parser.add_argument('--parallel', type=int, default=1,
                        help='Number of parallel processes (default: 1, sequential)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be done without actually running')
    parser.add_argument('--skip', type=str, default='',
                        help='Skip specific tasks (e.g., --skip 3,4)')
    
    args = parser.parse_args()
    
    # Validate task number
    if args.task != -1 and not (1 <= args.task <= 10):
        print(f"❌ Error: Task must be -1 or between 1-10, got {args.task}")
        sys.exit(1)
    
    # Find all videos
    print(f"\n🔍 Searching for videos in: {args.path}")
    videos = find_videos(args.path)
    
    if not videos:
        print(f"❌ No video files found in: {args.path}")
        print(f"   Supported formats: {', '.join(VIDEO_EXTENSIONS)}")
        sys.exit(1)
    
    print(f"\n📹 Found {len(videos)} video(s):")
    for i, video in enumerate(videos, 1):
        print(f"   {i}. {video}")
    
    if args.dry_run:
        print(f"\n🔍 Dry run mode - no actions performed")
        print(f"   Task: {'All tasks' if args.task == -1 else f'Task {args.task}'}")
        print(f"   ASR: {args.asr}")
        if args.skip:
            print(f"   Skipping tasks: {args.skip}")
        sys.exit(0)
    
    # Summary
    print(f"\n🚀 Processing Configuration:")
    print(f"   Total videos: {len(videos)}")
    print(f"   Task: {'All tasks' if args.task == -1 else f'Task {args.task}'}")
    print(f"   ASR: {args.asr}")
    print(f"   Parallel processes: {args.parallel}")
    if args.skip:
        print(f"   Skipping: {args.skip}")
    print()
    
    # Confirm before running
    confirm = input("Do you want to proceed? (y/n): ").strip().lower()
    if confirm != 'y':
        print("❌ Cancelled by user")
        sys.exit(0)
    
    # Process videos
    success_count = 0
    fail_count = 0
    
    if args.parallel > 1 and len(videos) > 1:
        # Parallel processing
        print(f"\n⚡ Running {args.parallel} parallel processes...")
        with ProcessPoolExecutor(max_workers=args.parallel) as executor:
            futures = {
                executor.submit(
                    process_video_with_retry, 
                    video, args.task, args.asr
                ): video for video in videos
            }
            
            for future in as_completed(futures):
                video_path, success, message = future.result()
                if success:
                    print(f"✅ Completed: {os.path.basename(video_path)}")
                    success_count += 1
                else:
                    print(f"❌ Failed: {os.path.basename(video_path)} - {message}")
                    fail_count += 1
    else:
        # Sequential processing
        print(f"\n🔄 Processing videos sequentially...")
        for i, video in enumerate(videos, 1):
            print(f"\n[{i}/{len(videos)}] ", end="")
            video_path, success, message = process_video_with_retry(video, args.task, args.asr)
            
            if success:
                print(f"✅ Completed: {os.path.basename(video_path)}")
                success_count += 1
            else:
                print(f"❌ Failed: {os.path.basename(video_path)} - {message}")
                fail_count += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"📊 Processing Complete!")
    print(f"   ✅ Success: {success_count}")
    print(f"   ❌ Failed: {fail_count}")
    print(f"   📁 Total: {len(videos)}")
    print(f"{'='*60}")
    
    if fail_count > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()


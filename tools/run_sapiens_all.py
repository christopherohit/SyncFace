#!/usr/bin/env python3
"""
Auto-run script for SyncFace Sapiens - Process all data folders with sapiens

Usage:
    python run_sapiens_all.py <path_to_data_folder>
    python run_sapiens_all.py data/pretrain/Jae-in/
    python run_sapiens_all.py data/pretrain/
    python run_sapiens_all.py .  # Process all data folders in current directory

Options:
    --type TYPE       Type of processing: depth, normal, or all (default: all)
    --gpu GPU_ID      GPU ID to use (default: 1 for second GPU)
    --dry-run         Show what would be done without actually running
    --parallel PAR    Number of parallel processes (default: 1)
"""

import os
import sys
import glob
import subprocess
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple
import time


def find_data_folders(path: str) -> List[str]:
    """
    Find all data folders that have gt_imgs directory (ready for sapiens processing).
    
    Args:
        path: Path to a data folder or parent directory
        
    Returns:
        List of absolute paths to data folders ready for processing
    """
    data_folders = []
    path = os.path.abspath(path)
    
    if os.path.isfile(path):
        # If it's a file, get its parent directory
        path = os.path.dirname(path)
    
    if os.path.isdir(path):
        # Check if this is already a data folder
        gt_imgs_path = os.path.join(path, 'gt_imgs')
        if os.path.isdir(gt_imgs_path):
            # Check if it has images
            images = glob.glob(os.path.join(gt_imgs_path, '*.jpg')) + \
                     glob.glob(os.path.join(gt_imgs_path, '*.png'))
            if images:
                data_folders.append(path)
        else:
            # Search recursively for gt_imgs folders
            for gt_imgs_dir in glob.glob(os.path.join(path, '**', 'gt_imgs'), recursive=True):
                data_folder = os.path.dirname(gt_imgs_dir)
                # Check if it has images
                images = glob.glob(os.path.join(gt_imgs_dir, '*.jpg')) + \
                         glob.glob(os.path.join(gt_imgs_dir, '*.png'))
                if images:
                    data_folders.append(data_folder)
    
    return sorted(set(data_folders))


def _run_sapiens_single(data_folder: str, script_type: str, gpu_id: int) -> Tuple[str, bool, str]:
    """Run a single sapiens script (depth or normal)."""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_dir)
        
        gt_imgs_path = os.path.join(data_folder, 'gt_imgs')
        
        if script_type == 'depth':
            script_path = 'data_utils_enhancement/sapiens/lite/scripts/depth.sh'
            script_name = 'depth'
        else:
            script_path = 'data_utils_enhancement/sapiens/lite/scripts/normal.sh'
            script_name = 'normal'
        
        # Modify the script to use the specified GPU
        script_content = []
        with open(script_path, 'r') as f:
            for line in f:
                if line.startswith('VALID_GPU_IDS='):
                    # Replace with single GPU ID
                    script_content.append(f'VALID_GPU_IDS=({gpu_id})\n')
                elif 'TOTAL_GPUS=' in line and 'VALID_GPU_IDS' not in line:
                    script_content.append('TOTAL_GPUS=1; VALID_GPU_IDS=(0)\n')
                else:
                    script_content.append(line)
        
        # Write modified script to temp file
        temp_script = f'/tmp/sapiens_{script_type}_{os.getpid()}.sh'
        with open(temp_script, 'w') as f:
            f.writelines(script_content)
        
        os.chmod(temp_script, 0o755)
        
        cmd = ['bash', temp_script, data_folder]
        
        print(f"\n{'='*60}")
        print(f"Processing: {data_folder}")
        print(f"Type: {script_name}, GPU: {gpu_id}")
        print(f"{'='*60}")
        
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        result = subprocess.run(
            cmd,
            capture_output=False,
            text=True,
            env=env
        )
        
        # Clean up temp script
        try:
            os.remove(temp_script)
        except:
            pass
        
        success = result.returncode == 0
        
        if success:
            return data_folder, True, f"{script_name} Success"
        else:
            return data_folder, False, f"{script_name} Failed with return code {result.returncode}"
            
    except Exception as e:
        return data_folder, False, f"Error: {str(e)}"


def run_sapiens_depth(data_folder: str, gpu_id: int = 1) -> Tuple[str, bool, str]:
    """Run only depth estimation."""
    return _run_sapiens_single(data_folder, 'depth', gpu_id)


def run_sapiens_normal(data_folder: str, gpu_id: int = 1) -> Tuple[str, bool, str]:
    """Run only normal estimation."""
    return _run_sapiens_single(data_folder, 'normal', gpu_id)


def run_sapiens_for_folder(data_folder: str, process_type: str = 'all', 
                           gpu_id: int = 1) -> Tuple[str, bool, str]:
    """
    Run sapiens processing for a single data folder.
    
    Args:
        data_folder: Path to the data folder (parent of gt_imgs)
        process_type: Type of processing (depth, normal, all)
        gpu_id: GPU ID to use
        
    Returns:
        Tuple of (folder_path, success, message)
    """
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_dir)
        
        gt_imgs_path = os.path.join(data_folder, 'gt_imgs')
        
        # Build the sapiens run command based on process_type
        if process_type == 'all':
            # Run both depth and normal
            depth_success = _run_sapiens_single(data_folder, 'depth', gpu_id)
            normal_success = _run_sapiens_single(data_folder, 'normal', gpu_id)
            
            if depth_success[1] and normal_success[1]:
                return data_folder, True, "All Success"
            elif depth_success[1]:
                return data_folder, False, "Depth Success, Normal Failed"
            elif normal_success[1]:
                return data_folder, False, "Depth Failed, Normal Success"
            else:
                return data_folder, False, "Both Failed"
        elif process_type == 'depth':
            return _run_sapiens_single(data_folder, 'depth', gpu_id)
        elif process_type == 'normal':
            return _run_sapiens_single(data_folder, 'normal', gpu_id)
        else:
            return data_folder, False, f"Unknown process type: {process_type}"
            
    except Exception as e:
        return data_folder, False, f"Error: {str(e)}"


def process_with_retry(data_folder: str, process_type: str, gpu_id: int,
                       max_retries: int = 2, retry_delay: int = 5) -> Tuple[str, bool, str]:
    """Process a data folder with retry logic."""
    for attempt in range(max_retries):
        folder_path, success, message = run_sapiens_for_folder(data_folder, process_type, gpu_id)
        
        if success:
            return folder_path, True, message
        
        if attempt < max_retries - 1:
            print(f"\n⚠️  Retrying in {retry_delay} seconds... (Attempt {attempt + 2}/{max_retries})")
            time.sleep(retry_delay)
    
    return data_folder, False, message


def main():
    parser = argparse.ArgumentParser(
        description='Auto-run SyncFace sapiens processing for all data folders',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process all data folders in a directory
    python run_sapiens_all.py data/pretrain/
    
    # Process a single data folder
    python run_sapiens_all.py data/pretrain/Jae-in/
    
    # Process only depth estimation
    python run_sapiens_all.py data/pretrain/ --type depth
    
    # Process only normal estimation
    python run_sapiens_all.py data/pretrain/ --type normal
    
    # Use GPU 0 (first GPU)
    python run_sapiens_all.py data/pretrain/ --gpu 0
    
    # Dry run - show what would be processed
    python run_sapiens_all.py data/pretrain/ --dry-run
    
    # Process with 2 parallel workers
    python run_sapiens_all.py data/pretrain/ --parallel 2
        """
    )
    
    parser.add_argument('path', type=str, nargs='?', default='.',
                        help='Path to data folder or parent directory (default: current directory)')
    parser.add_argument('--type', type=str, default='all',
                        choices=['depth', 'normal', 'all'],
                        help='Type of processing (default: all)')
    parser.add_argument('--gpu', type=int, default=1,
                        help='GPU ID to use (default: 1 for second GPU)')
    parser.add_argument('--parallel', type=int, default=1,
                        help='Number of parallel processes (default: 1)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be done without actually running')
    
    args = parser.parse_args()
    
    # Find all data folders
    print(f"\n🔍 Searching for data folders in: {args.path}")
    data_folders = find_data_folders(args.path)
    
    if not data_folders:
        print(f"❌ No data folders with gt_imgs found in: {args.path}")
        print("   Make sure the folder contains 'gt_imgs' directory with images")
        sys.exit(1)
    
    print(f"\n📁 Found {len(data_folders)} data folder(s) ready for sapiens processing:")
    for i, folder in enumerate(data_folders, 1):
        folder_name = os.path.basename(folder)
        gt_imgs_count = len(glob.glob(os.path.join(folder, 'gt_imgs', '*.jpg'))) + \
                       len(glob.glob(os.path.join(folder, 'gt_imgs', '*.png')))
        print(f"   {i}. {folder_name} ({gt_imgs_count} images)")
    
    if args.dry_run:
        print(f"\n🔍 Dry run mode - no actions performed")
        print(f"   Type: {args.type}")
        print(f"   GPU: {args.gpu}")
        sys.exit(0)
    
    # Summary
    print(f"\n🚀 Processing Configuration:")
    print(f"   Total data folders: {len(data_folders)}")
    print(f"   Processing type: {args.type}")
    print(f"   GPU ID: {args.gpu}")
    print(f"   Parallel processes: {args.parallel}")
    print()
    
    # Confirm before running
    confirm = input("Do you want to proceed? (y/n): ").strip().lower()
    if confirm != 'y':
        print("❌ Cancelled by user")
        sys.exit(0)
    
    # Process data folders
    success_count = 0
    fail_count = 0
    
    if args.parallel > 1 and len(data_folders) > 1:
        # Parallel processing
        print(f"\n⚡ Running {args.parallel} parallel processes...")
        with ProcessPoolExecutor(max_workers=args.parallel) as executor:
            futures = {
                executor.submit(
                    process_with_retry, 
                    folder, args.type, args.gpu
                ): folder for folder in data_folders
            }
            
            for future in as_completed(futures):
                folder_path, success, message = future.result()
                folder_name = os.path.basename(folder_path)
                if success:
                    print(f"✅ Completed: {folder_name}")
                    success_count += 1
                else:
                    print(f"❌ Failed: {folder_name} - {message}")
                    fail_count += 1
    else:
        # Sequential processing
        print(f"\n🔄 Processing data folders sequentially...")
        for i, folder in enumerate(data_folders, 1):
            print(f"\n[{i}/{len(data_folders)}] ", end="")
            folder_path, success, message = process_with_retry(folder, args.type, args.gpu)
            folder_name = os.path.basename(folder_path)
            
            if success:
                print(f"✅ Completed: {folder_name}")
                success_count += 1
            else:
                print(f"❌ Failed: {folder_name} - {message}")
                fail_count += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"📊 Sapiens Processing Complete!")
    print(f"   ✅ Success: {success_count}")
    print(f"   ❌ Failed: {fail_count}")
    print(f"   📁 Total: {len(data_folders)}")
    print(f"{'='*60}")
    
    if fail_count > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()


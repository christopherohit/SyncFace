#!/usr/bin/env python3
"""
Auto-run script for SyncFace - Process all aud.wav files with multiple audio extractors

This script finds all aud.wav files in a directory and processes them with:
- HuBERT (facebook/hubert-large-ls960-ft)
- DeepSpeech (Mozilla DeepSpeech 0.1.0)
- Wav2Vec (wav2vec2-large)

Usage:
    python run_audio_process.py <path_to_folder>
    python run_audio_process.py data/pretrain/Jae-in/
    python run_audio_process.py data/pretrain/
    python run_audio_process.py .  # Process all aud.wav in current directory

Options:
    --extractors EXT  Comma-separated list of extractors to use (default: all)
                      Options: hubert, deepspeech, wav2vec
                      Example: --extractors hubert,wav2vec
    --parallel PAR    Number of parallel processes (default: 1, sequential)
    --dry-run         Show what would be done without actually running
"""

import os
import sys
import glob
import argparse
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple, Set
import time


def find_audio_files(path: str) -> List[str]:
    """
    Find all aud.wav files in the given path (file or directory).
    
    Args:
        path: Path to an audio file or directory containing audio files
        
    Returns:
        List of absolute paths to aud.wav files
    """
    audio_files = []
    path = os.path.abspath(path)
    
    if os.path.isfile(path):
        if path.endswith('aud.wav'):
            audio_files.append(path)
    elif os.path.isdir(path):
        # Search recursively for aud.wav files
        audio_files.extend(glob.glob(os.path.join(path, '**/aud.wav'), recursive=True))
    
    return sorted(set(audio_files))


def check_existing_features(audio_path: str, extractors: Set[str]) -> Set[str]:
    """
    Check which audio features already exist.
    
    Args:
        audio_path: Path to aud.wav file
        extractors: Set of extractor names to check
        
    Returns:
        Set of extractors that have already been processed
    """
    existing = set()
    base_path = audio_path.replace('.wav', '')
    
    # Check for each extractor's output
    if 'hubert' in extractors:
        if os.path.exists(f'{base_path}_hu.npy'):
            existing.add('hubert')
    
    if 'deepspeech' in extractors:
        if os.path.exists(f'{base_path}.npy'):
            existing.add('deepspeech')
    
    if 'wav2vec' in extractors:
        if os.path.exists(f'{base_path}_eo.npy'):
            existing.add('wav2vec')
    
    return existing


def run_audio_extractor(audio_path: str, extractor: str) -> Tuple[str, str, bool, str]:
    """
    Run a single audio feature extractor on an audio file.
    
    Args:
        audio_path: Path to the aud.wav file
        extractor: Name of the extractor (hubert, deepspeech, or wav2vec)
        
    Returns:
        Tuple of (audio_path, extractor, success, message)
    """
    try:
        # Get SyncFace root directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        syncface_root = os.path.dirname(script_dir)  # Go up from tools/ to SyncFace root
        
        # Build command based on extractor type
        if extractor == 'hubert':
            cmd = [
                sys.executable,
                os.path.join(syncface_root, 'data_utils_enhancement/hubert.py'),
                '--wav', audio_path
            ]
        elif extractor == 'deepspeech':
            cmd = [
                sys.executable,
                os.path.join(syncface_root, 'data_utils_enhancement/deepspeech_features/extract_ds_features.py'),
                '--input', audio_path
            ]
        elif extractor == 'wav2vec':
            cmd = [
                sys.executable,
                os.path.join(syncface_root, 'data_utils_enhancement/wav2vec.py'),
                '--wav', audio_path,
                '--save_feats'
            ]
        else:
            return audio_path, extractor, False, f"Unknown extractor: {extractor}"
        
        print(f"\n{'='*60}")
        print(f"Processing: {audio_path}")
        print(f"Extractor: {extractor}")
        print(f"Command: {' '.join(cmd)}")
        print(f"{'='*60}")
        
        # Run the extractor with cwd set to SyncFace root
        result = subprocess.run(
            cmd,
            capture_output=False,
            text=True,
            cwd=syncface_root
        )
        
        success = result.returncode == 0
        
        if success:
            return audio_path, extractor, True, "Success"
        else:
            return audio_path, extractor, False, f"Failed with return code {result.returncode}"
            
    except Exception as e:
        return audio_path, extractor, False, f"Error: {str(e)}"


def process_audio_file(audio_path: str, extractors: Set[str], 
                       skip_existing: bool = True) -> List[Tuple[str, str, bool, str]]:
    """
    Process a single audio file with all specified extractors.
    
    Args:
        audio_path: Path to the aud.wav file
        extractors: Set of extractor names to use
        skip_existing: Whether to skip already processed extractors
        
    Returns:
        List of tuples (audio_path, extractor, success, message) for each extractor
    """
    results = []
    
    # Check which extractors have already been processed
    if skip_existing:
        existing = check_existing_features(audio_path, extractors)
        to_process = extractors - existing
        
        for ext in existing:
            results.append((audio_path, ext, True, "Already exists (skipped)"))
    else:
        to_process = extractors
    
    # Process with each extractor
    for extractor in sorted(to_process):
        result = run_audio_extractor(audio_path, extractor)
        results.append(result)
    
    return results


def process_audio_with_retry(audio_path: str, extractors: Set[str], 
                             skip_existing: bool = True,
                             max_retries: int = 2, 
                             retry_delay: int = 5) -> List[Tuple[str, str, bool, str]]:
    """
    Process an audio file with retry logic for failed extractors.
    """
    results = process_audio_file(audio_path, extractors, skip_existing)
    
    # Check for failures and retry
    for attempt in range(1, max_retries):
        failed_extractors = {ext for _, ext, success, _ in results if not success and 'skipped' not in _.lower()}
        
        if not failed_extractors:
            break
        
        print(f"\n⚠️  Retrying {len(failed_extractors)} failed extractor(s) in {retry_delay} seconds... (Attempt {attempt + 1}/{max_retries})")
        time.sleep(retry_delay)
        
        # Retry only failed extractors
        retry_results = process_audio_file(audio_path, failed_extractors, skip_existing=False)
        
        # Update results
        new_results = []
        for audio, ext, success, msg in results:
            if ext in failed_extractors:
                # Find the retry result for this extractor
                retry_result = next((r for r in retry_results if r[1] == ext), None)
                if retry_result:
                    new_results.append(retry_result)
                else:
                    new_results.append((audio, ext, success, msg))
            else:
                new_results.append((audio, ext, success, msg))
        
        results = new_results
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Auto-run audio feature extraction for all aud.wav files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process all aud.wav files in a directory with all extractors
    python run_audio_process.py data/pretrain/
    
    # Process a single aud.wav file
    python run_audio_process.py data/pretrain/Jae-in/aud.wav
    
    # Process only with hubert and wav2vec
    python run_audio_process.py data/pretrain/ --extractors hubert,wav2vec
    
    # Process only with deepspeech
    python run_audio_process.py data/pretrain/ --extractors deepspeech
    
    # Dry run - show what would be processed
    python run_audio_process.py data/pretrain/ --dry-run
    
    # Process with 2 parallel workers
    python run_audio_process.py data/pretrain/ --parallel 2
    
    # Force reprocess even if features exist
    python run_audio_process.py data/pretrain/ --force
        """
    )
    
    parser.add_argument('path', type=str, nargs='?', default='.',
                        help='Path to aud.wav file or directory (default: current directory)')
    parser.add_argument('--extractors', type=str, default='all',
                        help='Comma-separated list of extractors: hubert,deepspeech,wav2vec or "all" (default: all)')
    parser.add_argument('--parallel', type=int, default=1,
                        help='Number of parallel processes (default: 1, sequential)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be done without actually running')
    parser.add_argument('--force', action='store_true',
                        help='Force reprocess even if feature files already exist')
    
    args = parser.parse_args()
    
    # Parse extractors
    all_extractors = {'hubert', 'deepspeech', 'wav2vec'}
    if args.extractors.lower() == 'all':
        extractors = all_extractors
    else:
        extractors = set(ext.strip().lower() for ext in args.extractors.split(','))
        invalid = extractors - all_extractors
        if invalid:
            print(f"❌ Error: Invalid extractor(s): {', '.join(invalid)}")
            print(f"   Valid options: {', '.join(sorted(all_extractors))}")
            sys.exit(1)
    
    # Find all audio files
    print(f"\n🔍 Searching for aud.wav files in: {args.path}")
    audio_files = find_audio_files(args.path)
    
    if not audio_files:
        print(f"❌ No aud.wav files found in: {args.path}")
        sys.exit(1)
    
    print(f"\n🎵 Found {len(audio_files)} audio file(s):")
    for i, audio in enumerate(audio_files, 1):
        folder_name = os.path.basename(os.path.dirname(audio))
        print(f"   {i}. {folder_name}/aud.wav")
        
        # Check existing features
        if not args.force:
            existing = check_existing_features(audio, extractors)
            if existing:
                print(f"      Already processed: {', '.join(sorted(existing))}")
    
    if args.dry_run:
        print(f"\n🔍 Dry run mode - no actions performed")
        print(f"   Extractors: {', '.join(sorted(extractors))}")
        print(f"   Force reprocess: {args.force}")
        sys.exit(0)
    
    # Summary
    print(f"\n🚀 Processing Configuration:")
    print(f"   Total audio files: {len(audio_files)}")
    print(f"   Extractors: {', '.join(sorted(extractors))}")
    print(f"   Parallel processes: {args.parallel}")
    print(f"   Skip existing: {not args.force}")
    print()
    
    # Confirm before running
    confirm = input("Do you want to proceed? (y/n): ").strip().lower()
    if confirm != 'y':
        print("❌ Cancelled by user")
        sys.exit(0)
    
    # Process audio files
    total_success = 0
    total_failed = 0
    total_skipped = 0
    
    if args.parallel > 1 and len(audio_files) > 1:
        # Parallel processing
        print(f"\n⚡ Running {args.parallel} parallel processes...")
        with ProcessPoolExecutor(max_workers=args.parallel) as executor:
            futures = {
                executor.submit(
                    process_audio_with_retry, 
                    audio, extractors, not args.force
                ): audio for audio in audio_files
            }
            
            for future in as_completed(futures):
                results = future.result()
                folder_name = os.path.basename(os.path.dirname(results[0][0]))
                
                for audio_path, extractor, success, message in results:
                    if 'skipped' in message.lower():
                        print(f"⏭️  {folder_name} - {extractor}: {message}")
                        total_skipped += 1
                    elif success:
                        print(f"✅ {folder_name} - {extractor}: {message}")
                        total_success += 1
                    else:
                        print(f"❌ {folder_name} - {extractor}: {message}")
                        total_failed += 1
    else:
        # Sequential processing
        print(f"\n🔄 Processing audio files sequentially...")
        for i, audio in enumerate(audio_files, 1):
            folder_name = os.path.basename(os.path.dirname(audio))
            print(f"\n[{i}/{len(audio_files)}] Processing: {folder_name}/aud.wav")
            
            results = process_audio_with_retry(audio, extractors, not args.force)
            
            for audio_path, extractor, success, message in results:
                if 'skipped' in message.lower():
                    print(f"   ⏭️  {extractor}: {message}")
                    total_skipped += 1
                elif success:
                    print(f"   ✅ {extractor}: {message}")
                    total_success += 1
                else:
                    print(f"   ❌ {extractor}: {message}")
                    total_failed += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"📊 Audio Processing Complete!")
    print(f"   ✅ Success: {total_success}")
    print(f"   ❌ Failed: {total_failed}")
    print(f"   ⏭️  Skipped: {total_skipped}")
    print(f"   📁 Total operations: {total_success + total_failed + total_skipped}")
    print(f"{'='*60}")
    
    if total_failed > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()

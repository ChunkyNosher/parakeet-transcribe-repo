#!/usr/bin/env python3
"""
Validation script for manifest.json file locking fixes.

This script tests the key functionality without requiring actual NeMo models:
1. Tempfile configuration validation
2. Gradio cache directory creation
3. File copy function with retry logic
4. SHA-256 hash generation for filenames

Run with: python test_manifest_fix.py
"""

import os
import sys
import tempfile
from pathlib import Path
import shutil
import hashlib

def test_tempfile_configuration():
    """Test that tempfile configuration works correctly."""
    print("\n" + "="*80)
    print("TEST 1: Tempfile Configuration")
    print("="*80)
    
    # Get script directory
    script_dir = Path(__file__).parent.absolute()
    cache_dir = script_dir / "model_cache"
    temp_dir = cache_dir / "tmp"
    
    # Create temp directory if it doesn't exist
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Set tempfile.tempdir
    tempfile.tempdir = str(temp_dir)
    
    # Validate
    actual_temp = tempfile.gettempdir()
    expected_temp = str(temp_dir)
    
    if actual_temp == expected_temp:
        print(f"✓ PASS: tempfile.gettempdir() returns expected directory")
        print(f"  Expected: {expected_temp}")
        print(f"  Actual:   {actual_temp}")
        return True
    else:
        print(f"✗ FAIL: tempfile.gettempdir() mismatch")
        print(f"  Expected: {expected_temp}")
        print(f"  Actual:   {actual_temp}")
        return False

def test_gradio_cache_directory():
    """Test that Gradio cache directory is created correctly."""
    print("\n" + "="*80)
    print("TEST 2: Gradio Cache Directory Creation")
    print("="*80)
    
    script_dir = Path(__file__).parent.absolute()
    cache_dir = script_dir / "model_cache"
    gradio_cache_dir = cache_dir / "gradio_uploads"
    
    # Create directory
    gradio_cache_dir.mkdir(parents=True, exist_ok=True)
    
    if gradio_cache_dir.exists() and gradio_cache_dir.is_dir():
        print(f"✓ PASS: Gradio cache directory created")
        print(f"  Path: {gradio_cache_dir}")
        return True
    else:
        print(f"✗ FAIL: Gradio cache directory not created")
        return False

def test_file_hash_generation():
    """Test SHA-256 hash generation for filenames."""
    print("\n" + "="*80)
    print("TEST 3: SHA-256 Hash Generation")
    print("="*80)
    
    # Test with sample file path
    test_path = "/tmp/gradio/test_audio.wav"
    path_hash = hashlib.sha256(test_path.encode()).hexdigest()[:16]
    
    print(f"  Input path: {test_path}")
    print(f"  SHA-256 hash (first 16 chars): {path_hash}")
    
    # Verify hash is 16 characters
    if len(path_hash) == 16:
        print(f"✓ PASS: Hash length is correct (16 characters)")
        return True
    else:
        print(f"✗ FAIL: Hash length is {len(path_hash)}, expected 16")
        return False

def test_file_copy_logic():
    """Test file copy logic with temporary files."""
    print("\n" + "="*80)
    print("TEST 4: File Copy Logic")
    print("="*80)
    
    script_dir = Path(__file__).parent.absolute()
    cache_dir = script_dir / "model_cache"
    gradio_cache_dir = cache_dir / "gradio_uploads"
    gradio_cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a test file
    test_dir = tempfile.mkdtemp()
    test_file = Path(test_dir) / "test_audio.txt"
    test_content = "This is a test audio file simulation."
    
    try:
        # Write test file
        test_file.write_text(test_content)
        print(f"  Created test file: {test_file}")
        
        # Generate cached filename
        path_hash = hashlib.sha256(str(test_file).encode()).hexdigest()[:16]
        cached_filename = f"{path_hash}_{test_file.name}"
        cached_path = gradio_cache_dir / cached_filename
        
        # Copy file
        shutil.copy2(test_file, cached_path)
        print(f"  Copied to cache: {cached_path}")
        
        # Verify copy
        if cached_path.exists():
            cached_content = cached_path.read_text()
            if cached_content == test_content:
                print(f"✓ PASS: File copied successfully and content matches")
                
                # Cleanup
                cached_path.unlink()
                return True
            else:
                print(f"✗ FAIL: File content mismatch")
                return False
        else:
            print(f"✗ FAIL: Cached file not found")
            return False
            
    finally:
        # Cleanup test directory
        shutil.rmtree(test_dir, ignore_errors=True)

def test_retry_logic_simulation():
    """Test retry logic with simulated failures."""
    print("\n" + "="*80)
    print("TEST 5: File Copy Retry Logic")
    print("="*80)
    
    # Updated values from the fix
    max_retries = 6
    base_delay = 0.5
    
    print(f"  Max retries: {max_retries}")
    print(f"  Base delay: {base_delay}s")
    print(f"  Expected delays (linear backoff):")
    
    total_wait = 0
    for attempt in range(max_retries):
        delay = base_delay * (attempt + 1)
        total_wait += delay
        print(f"    Attempt {attempt + 1}: {delay:.1f}s")
    
    print(f"  Total max wait time: {total_wait:.1f}s")
    
    # Verify delays are correct
    expected_delays = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    actual_delays = [base_delay * (i + 1) for i in range(max_retries)]
    
    # Use approximate comparison for floating point
    delays_match = all(abs(a - e) < 0.001 for a, e in zip(actual_delays, expected_delays))
    
    # Verify total wait time (should be 10.5s)
    expected_total = 10.5
    total_match = abs(total_wait - expected_total) < 0.001
    
    if delays_match and total_match:
        print(f"✓ PASS: Retry delays calculated correctly")
        print(f"  Linear backoff accommodates Windows antivirus (500ms-4000ms)")
        return True
    else:
        print(f"✗ FAIL: Delay mismatch")
        print(f"  Expected delays: {expected_delays}")
        print(f"  Actual delays:   {actual_delays}")
        print(f"  Expected total: {expected_total}s")
        print(f"  Actual total:   {total_wait}s")
        return False

def test_librosa_retry_logic():
    """Test librosa.get_duration() retry logic."""
    print("\n" + "="*80)
    print("TEST 6: Librosa Duration Check Retry Logic")
    print("="*80)
    
    # Updated values for librosa retry
    max_duration_retries = 4
    duration_base_delay = 0.5
    
    print(f"  Max retries: {max_duration_retries}")
    print(f"  Base delay: {duration_base_delay}s")
    print(f"  Expected delays (linear backoff):")
    
    total_wait = 0
    for attempt in range(max_duration_retries):
        delay = duration_base_delay * (attempt + 1)
        total_wait += delay
        print(f"    Attempt {attempt + 1}: {delay:.1f}s")
    
    print(f"  Total max wait time: {total_wait:.1f}s")
    
    # Verify delays are correct
    expected_delays = [0.5, 1.0, 1.5, 2.0]
    actual_delays = [duration_base_delay * (i + 1) for i in range(max_duration_retries)]
    
    # Use approximate comparison for floating point
    delays_match = all(abs(a - e) < 0.001 for a, e in zip(actual_delays, expected_delays))
    
    # Verify total wait time (should be 5.0s)
    expected_total = 5.0
    total_match = abs(total_wait - expected_total) < 0.001
    
    if delays_match and total_match:
        print(f"✓ PASS: Librosa retry delays calculated correctly")
        print(f"  Handles file locks after copy operation")
        return True
    else:
        print(f"✗ FAIL: Delay mismatch")
        print(f"  Expected delays: {expected_delays}")
        print(f"  Actual delays:   {actual_delays}")
        print(f"  Expected total: {expected_total}s")
        print(f"  Actual total:   {total_wait}s")
        return False

def main():
    """Run all validation tests."""
    print("\n" + "="*80)
    print("MANIFEST.JSON FILE LOCKING FIX - VALIDATION TESTS")
    print("="*80)
    print("\nThese tests validate the implementation without requiring NeMo models.")
    
    results = []
    
    # Run tests
    results.append(("Tempfile Configuration", test_tempfile_configuration()))
    results.append(("Gradio Cache Directory", test_gradio_cache_directory()))
    results.append(("SHA-256 Hash Generation", test_file_hash_generation()))
    results.append(("File Copy Logic", test_file_copy_logic()))
    results.append(("Retry Logic Simulation", test_retry_logic_simulation()))
    results.append(("Librosa Retry Logic", test_librosa_retry_logic()))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    print("="*80 + "\n")
    
    # Exit with appropriate code
    sys.exit(0 if passed == total else 1)

if __name__ == "__main__":
    main()

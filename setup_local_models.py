#!/usr/bin/env python3
"""
CORRECTED: One-time setup script to download and save NeMo ASR models locally.
Includes fixes for cache corruption and Windows path issues.
"""

import nemo.collections.asr as nemo_asr
import os
import sys
import shutil

def create_local_models_directory():
    """Create local_models directory if it doesn't exist."""
    if not os.path.exists("local_models"):
        os.makedirs("local_models")
        print("✓ Created local_models/ directory")
    else:
        print("✓ local_models/ directory already exists")

def clear_model_cache():
    """Clear NeMo cache to force fresh download."""
    cache_paths = [
        os.path.expanduser("~/.cache/torch/NeMo"),
        os.path.expanduser("~/.cache/huggingface"),
        os.path.join(os.path.expanduser("~"), ".cache", "torch", "NeMo"),
    ]
    
    print("\n🧹 Clearing model cache to ensure clean download...")
    for cache_path in cache_paths:
        if os.path.exists(cache_path):
            try:
                shutil.rmtree(cache_path)
                print(f"✓ Cleared: {cache_path}")
            except Exception as e:
                print(f"⚠️  Could not clear {cache_path}: {e}")
    print("✓ Cache cleared\n")

def download_and_save_parakeet():
    """Download Parakeet model and save as .nemo file."""
    print("\n" + "="*80)
    print("📦 Downloading Parakeet-TDT-0.6B v2 from HuggingFace...")
    print("="*80)
    print("This will take ~2-5 minutes depending on your connection.")
    print("Model size: ~1.2 GB\n")
    
    try:
        # Download with explicit cache clearing
        model = nemo_asr.models.ASRModel.from_pretrained(
            "nvidia/parakeet-tdt-0.6b-v2",
            refresh_cache=True  # Force fresh download
        )
        print("\n✓ Parakeet downloaded successfully")
        
        # Save to local file
        print("\n💾 Saving Parakeet to local_models/parakeet.nemo...")
        output_path = os.path.abspath("local_models/parakeet.nemo")
        model.save_to(output_path)
        print(f"✓ Parakeet saved successfully to {output_path}")
        
        # Verify file exists and has reasonable size
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path) / (1024**3)
            print(f"✓ Verified: parakeet.nemo ({file_size:.2f} GB)")
            
            if file_size < 0.5:
                print(f"⚠️  Warning: File size seems small ({file_size:.2f} GB)")
                return False
        else:
            print("❌ File was not created!")
            return False
        
        del model  # Free memory before next download
        return True
        
    except Exception as e:
        print(f"\n❌ Error downloading/saving Parakeet:")
        print(f"   {type(e).__name__}: {str(e)}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        return False

def download_and_save_canary():
    """Download Canary model and save as .nemo file."""
    print("\n" + "="*80)
    print("📦 Downloading Canary-Qwen-2.5B from HuggingFace...")
    print("="*80)
    print("This will take ~10-20 minutes depending on your connection.")
    print("Model size: ~5.3 GB\n")
    
    try:
        # Download with explicit cache clearing
        model = nemo_asr.models.ASRModel.from_pretrained(
            "nvidia/canary-qwen-2.5b",
            refresh_cache=True  # Force fresh download
        )
        print("\n✓ Canary downloaded successfully")
        
        # Save to local file with absolute path
        print("\n💾 Saving Canary to local_models/canary.nemo...")
        output_path = os.path.abspath("local_models/canary.nemo")
        model.save_to(output_path)
        print(f"✓ Canary saved successfully to {output_path}")
        
        # Verify file exists and has reasonable size
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path) / (1024**3)
            print(f"✓ Verified: canary.nemo ({file_size:.2f} GB)")
            
            if file_size < 2.0:
                print(f"⚠️  Warning: File size seems small ({file_size:.2f} GB)")
                return False
        else:
            print("❌ File was not created!")
            return False
        
        del model  # Free memory
        return True
        
    except Exception as e:
        print(f"\n❌ Error downloading/saving Canary:")
        print(f"   {type(e).__name__}: {str(e)}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        return False

def verify_setup():
    """Verify both .nemo files exist and are valid."""
    print("\n" + "="*80)
    print("🔍 Verifying Setup...")
    print("="*80)
    
    errors = []
    
    # Check Parakeet
    parakeet_path = os.path.abspath("local_models/parakeet.nemo")
    if not os.path.exists(parakeet_path):
        errors.append(f"❌ parakeet.nemo not found at {parakeet_path}")
    else:
        size = os.path.getsize(parakeet_path) / (1024**3)
        if size < 0.5:
            errors.append(f"⚠️  parakeet.nemo too small ({size:.2f} GB), may be corrupted")
        else:
            print(f"✓ parakeet.nemo exists ({size:.2f} GB)")
    
    # Check Canary
    canary_path = os.path.abspath("local_models/canary.nemo")
    if not os.path.exists(canary_path):
        errors.append(f"❌ canary.nemo not found at {canary_path}")
    else:
        size = os.path.getsize(canary_path) / (1024**3)
        if size < 2.0:
            errors.append(f"⚠️  canary.nemo too small ({size:.2f} GB), may be corrupted")
        else:
            print(f"✓ canary.nemo exists ({size:.2f} GB)")
    
    if errors:
        print("\n⚠️  Setup incomplete:")
        for error in errors:
            print(f"  {error}")
        return False
    else:
        print("\n✅ All models ready!")
        return True

def main():
    print("\n" + "="*80)
    print("🚀 NeMo ASR Local Model Setup (CORRECTED VERSION)")
    print("="*80)
    print("\nThis script will:")
    print("  1. Clear existing NeMo cache (to fix corruption)")
    print("  2. Create local_models/ directory")
    print("  3. Download Parakeet-TDT-0.6B v2 (~1.2 GB)")
    print("  4. Download Canary-Qwen-2.5B (~5.3 GB)")
    print("  5. Save both as .nemo files for offline use")
    print("\nTotal download: ~6.5 GB")
    print("Estimated time: 20-30 minutes\n")
    
    response = input("Continue with setup? (y/n): ").strip().lower()
    if response != 'y':
        print("\n❌ Setup cancelled")
        sys.exit(0)
    
    # Clear cache first to avoid corruption
    clear_model_cache()
    
    # Create directory
    create_local_models_directory()
    
    # Download and save Parakeet
    print("\n" + "="*80)
    print("STEP 1/2: PARAKEET")
    print("="*80)
    parakeet_success = download_and_save_parakeet()
    
    if not parakeet_success:
        print("\n❌ Parakeet setup failed.")
        print("\nTroubleshooting:")
        print("1. Check your internet connection")
        print("2. Ensure you have at least 5GB free disk space")
        print("3. Try running as administrator if on Windows")
        print("4. Check antivirus isn't blocking file creation")
        sys.exit(1)
    
    # Download and save Canary
    print("\n" + "="*80)
    print("STEP 2/2: CANARY")
    print("="*80)
    canary_success = download_and_save_canary()
    
    if not canary_success:
        print("\n❌ Canary setup failed.")
        print("\nTroubleshooting:")
        print("1. Check your internet connection")
        print("2. Ensure you have at least 10GB free disk space")
        print("3. Try running as administrator if on Windows")
        print("4. Check antivirus isn't blocking file creation")
        sys.exit(1)
    
    # Verify everything
    if verify_setup():
        print("\n" + "="*80)
        print("✅ Setup Complete!")
        print("="*80)
        print("\nYou can now run transcribe_ui.py")
        print("Models will load from local files (no internet required)")
        print("\nLocal model files:")
        print(f"  • {os.path.abspath('local_models/parakeet.nemo')}")
        print(f"  • {os.path.abspath('local_models/canary.nemo')}")
        print("\n⚠️  Keep these files - deleting them will require re-download")
        print("="*80 + "\n")
    else:
        print("\n❌ Setup verification failed")
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

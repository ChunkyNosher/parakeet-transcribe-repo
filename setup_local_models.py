#!/usr/bin/env python3
"""
One-time setup script to download and save NeMo ASR models locally.
This eliminates HuggingFace cache issues and enables offline operation.
"""

import nemo.collections.asr as nemo_asr
import os
import sys

def create_local_models_directory():
    """Create local_models directory if it doesn't exist."""
    if not os.path.exists("local_models"):
        os.makedirs("local_models")
        print("✓ Created local_models/ directory")
    else:
        print("✓ local_models/ directory already exists")

def download_and_save_parakeet():
    """Download Parakeet model and save as .nemo file."""
    print("\n" + "="*80)
    print("📦 Downloading Parakeet-TDT-0.6B v2 from HuggingFace...")
    print("="*80)
    print("This will take ~2-5 minutes depending on your connection.")
    print("Model size: ~1.2 GB\n")
    
    try:
        model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v2")
        print("\n✓ Parakeet downloaded successfully")
        
        print("\n💾 Saving Parakeet to local_models/parakeet.nemo...")
        model.save_to("local_models/parakeet.nemo")
        print("✓ Parakeet saved successfully")
        
        # Verify file exists and has reasonable size
        file_size = os.path.getsize("local_models/parakeet.nemo") / (1024**3)
        print(f"✓ Verified: parakeet.nemo ({file_size:.2f} GB)")
        
        del model  # Free memory before next download
        return True
        
    except Exception as e:
        print(f"\n❌ Error downloading/saving Parakeet: {e}")
        return False

def download_and_save_canary():
    """Download Canary model and save as .nemo file."""
    print("\n" + "="*80)
    print("📦 Downloading Canary-Qwen-2.5B from HuggingFace...")
    print("="*80)
    print("This will take ~10-20 minutes depending on your connection.")
    print("Model size: ~5.3 GB\n")
    
    try:
        model = nemo_asr.models.ASRModel.from_pretrained("nvidia/canary-qwen-2.5b")
        print("\n✓ Canary downloaded successfully")
        
        print("\n💾 Saving Canary to local_models/canary.nemo...")
        model.save_to("local_models/canary.nemo")
        print("✓ Canary saved successfully")
        
        # Verify file exists and has reasonable size
        file_size = os.path.getsize("local_models/canary.nemo") / (1024**3)
        print(f"✓ Verified: canary.nemo ({file_size:.2f} GB)")
        
        del model  # Free memory
        return True
        
    except Exception as e:
        print(f"\n❌ Error downloading/saving Canary: {e}")
        return False

def verify_setup():
    """Verify both .nemo files exist and are valid."""
    print("\n" + "="*80)
    print("🔍 Verifying Setup...")
    print("="*80)
    
    errors = []
    
    # Check Parakeet
    if not os.path.exists("local_models/parakeet.nemo"):
        errors.append("❌ parakeet.nemo not found")
    else:
        size = os.path.getsize("local_models/parakeet.nemo") / (1024**3)
        if size < 0.5:  # Should be ~1.2GB
            errors.append(f"⚠️  parakeet.nemo too small ({size:.2f} GB), may be corrupted")
        else:
            print(f"✓ parakeet.nemo exists ({size:.2f} GB)")
    
    # Check Canary
    if not os.path.exists("local_models/canary.nemo"):
        errors.append("❌ canary.nemo not found")
    else:
        size = os.path.getsize("local_models/canary.nemo") / (1024**3)
        if size < 2.0:  # Should be ~5.3GB
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
    print("🚀 NeMo ASR Local Model Setup")
    print("="*80)
    print("\nThis script will:")
    print("  1. Create local_models/ directory")
    print("  2. Download Parakeet-TDT-0.6B v2 (~1.2 GB)")
    print("  3. Download Canary-Qwen-2.5B (~5.3 GB)")
    print("  4. Save both as .nemo files for offline use")
    print("\nTotal download: ~6.5 GB")
    print("Estimated time: 15-25 minutes\n")
    
    response = input("Continue with setup? (y/n): ").strip().lower()
    if response != 'y':
        print("\n❌ Setup cancelled")
        sys.exit(0)
    
    # Create directory
    create_local_models_directory()
    
    # Download and save Parakeet
    parakeet_success = download_and_save_parakeet()
    if not parakeet_success:
        print("\n❌ Parakeet setup failed. Please check error and try again.")
        sys.exit(1)
    
    # Download and save Canary
    canary_success = download_and_save_canary()
    if not canary_success:
        print("\n❌ Canary setup failed. Please check error and try again.")
        sys.exit(1)
    
    # Verify everything
    if verify_setup():
        print("\n" + "="*80)
        print("✅ Setup Complete!")
        print("="*80)
        print("\nYou can now run transcribe_ui.py")
        print("Models will load from local files (no internet required)")
        print("\nLocal model files:")
        print("  • local_models/parakeet.nemo")
        print("  • local_models/canary.nemo")
        print("\n⚠️  Keep these files - deleting them will require re-download")
        print("="*80 + "\n")
    else:
        print("\n❌ Setup verification failed")
        sys.exit(1)

if __name__ == "__main__":
    main()

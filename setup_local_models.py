#!/usr/bin/env python3
"""
Simplified setup: Downloads and saves Parakeet models as .nemo files.
Canary models load automatically from HuggingFace on first use.

Available Parakeet models:
- Parakeet-TDT-0.6B-v3: 600M params, 25 languages, multilingual (RECOMMENDED)
- Parakeet-TDT-1.1B: 1.1B params, English only, best accuracy (1.5% WER)

Both models can be saved to local .nemo files for faster offline loading.
Canary models (Canary-1B, Canary-1B-v2) download automatically on first use.

See docs/manual/comprehensive-asr-model-integration-guide.md for details.
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
    """Download and save a Parakeet model as .nemo file."""
    print("\n" + "="*80)
    print("📦 PARAKEET MODEL SELECTION")
    print("="*80)
    print("\nAvailable models:")
    print("\n1. Parakeet-TDT-0.6B-v3 (RECOMMENDED)")
    print("   - Size: ~1.2 GB download, ~2.4 GB saved")
    print("   - Languages: 25 European languages with auto-detection")
    print("   - Speed: 3,380× real-time (ultra-fast)")
    print("   - Accuracy: ~1.7% WER")
    print("   - Best for: Multilingual transcription, general use")
    
    print("\n2. Parakeet-TDT-1.1B")
    print("   - Size: ~2.2 GB download, ~4.5 GB saved")
    print("   - Languages: English only")
    print("   - Speed: 1,336× real-time")
    print("   - Accuracy: 1.5% WER (BEST available)")
    print("   - Best for: Maximum English transcription accuracy")
    
    print("\n" + "="*80)
    choice = input("\nSelect model to download (1 or 2): ").strip()
    
    if choice == "1":
        model_id = "nvidia/parakeet-tdt-0.6b-v3"
        model_name = "Parakeet-TDT-0.6B-v3"
        download_size = "~1.2 GB"
        saved_size = "~2.4 GB"
    elif choice == "2":
        model_id = "nvidia/parakeet-tdt-1.1b"
        model_name = "Parakeet-TDT-1.1B"
        download_size = "~2.2 GB"
        saved_size = "~4.5 GB"
    else:
        print("\n❌ Invalid choice. Please run again and select 1 or 2.")
        sys.exit(1)
    
    print("\n" + "="*80)
    print(f"📦 Downloading {model_name} from HuggingFace...")
    print("="*80)
    print(f"Download size: {download_size}")
    print(f"Saved size: {saved_size}")
    print("This will take ~2-5 minutes depending on your connection.\n")
    
    try:
        # Download from HuggingFace
        model = nemo_asr.models.ASRModel.from_pretrained(model_id)
        print(f"\n✓ {model_name} downloaded successfully")
        
        # Save to local file
        print(f"\n💾 Saving {model_name} to local_models/parakeet.nemo...")
        output_path = os.path.abspath("local_models/parakeet.nemo")
        model.save_to(output_path)
        print(f"✓ {model_name} saved successfully to {output_path}")
        
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
        
        del model  # Free memory
        return True
        
    except Exception as e:
        print(f"\n❌ Error downloading/saving {model_name}:")
        print(f"   {type(e).__name__}: {str(e)}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        return False


def verify_setup():
    """Verify Parakeet .nemo file exists and is valid."""
    print("\n" + "="*80)
    print("🔍 Verifying Setup...")
    print("="*80)
    
    # Check Parakeet
    parakeet_path = os.path.abspath("local_models/parakeet.nemo")
    if not os.path.exists(parakeet_path):
        print(f"❌ parakeet.nemo not found at {parakeet_path}")
        return False
    
    size = os.path.getsize(parakeet_path) / (1024**3)
    if size < 0.5:
        print(f"⚠️  parakeet.nemo too small ({size:.2f} GB), may be corrupted")
        return False
    
    print(f"✓ parakeet.nemo exists ({size:.2f} GB)")
    print("\n✅ Parakeet setup complete!")
    return True


def main():
    print("\n" + "="*80)
    print("📦 NeMo ASR Local Model Setup")
    print("="*80)
    print("\nThis script will:")
    print("  1. Create local_models/ directory")
    print("  2. Download your selected Parakeet model")
    print("  3. Save as local .nemo file")
    print("\n⚠️  Note about other models:")
    print("  - Canary-1B and Canary-1B-v2 download automatically from HuggingFace")
    print("  - They will be cached for offline use after first download")
    print("  - No setup script needed for Canary models")
    print("\nEstimated time: 5-10 minutes\n")
    
    response = input("Continue with Parakeet setup? (y/n): ").strip().lower()
    if response != 'y':
        print("\n❌ Setup cancelled")
        sys.exit(0)
    
    # Create directory
    create_local_models_directory()
    
    # Download and save Parakeet
    print("\n" + "="*80)
    print("DOWNLOADING PARAKEET")
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
    
    # Verify
    if verify_setup():
        print("\n" + "="*80)
        print("✅ Setup Complete!")
        print("="*80)
        print("\nYou can now run transcribe_ui.py")
        print("\nModel loading behavior:")
        print(f"  • Parakeet: Loads from {os.path.abspath('local_models/parakeet.nemo')}")
        print("              (instant, works offline)")
        print("  • Canary models: Download from HuggingFace on first use")
        print("                   (then cached for offline use)")
        print("\n⚠️  Keep the local_models folder - deleting it requires re-download")
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

#!/usr/bin/env python3
"""
Simplified setup: Only downloads and saves Parakeet as .nemo file.
Canary uses SALM architecture and will load from HuggingFace cache automatically.

Why only Parakeet?
- Canary-Qwen-2.5B is a SALM (Speech-Aware Language Model) that cannot be 
  saved as a .nemo file due to its architecture
- Canary will download to HuggingFace cache on first use (~5GB)
- Once cached, Canary works offline like Parakeet

See docs/manual/canary-hybrid-loading-fix.md for technical details.
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
    print("Model size: ~1.2 GB download, ~2.4 GB saved\n")
    
    try:
        # Download from HuggingFace
        model = nemo_asr.models.ASRModel.from_pretrained(
            "nvidia/parakeet-tdt-0.6b-v2"
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
        
        del model  # Free memory
        return True
        
    except Exception as e:
        print(f"\n❌ Error downloading/saving Parakeet:")
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
    print("📦 NeMo ASR Local Model Setup (Parakeet Only)")
    print("="*80)
    print("\nThis script will:")
    print("  1. Create local_models/ directory")
    print("  2. Download Parakeet-TDT-0.6B v2 (~1.2 GB)")
    print("  3. Save as local .nemo file (~2.4 GB)")
    print("\n⚠️  Note about Canary:")
    print("  Canary-Qwen-2.5B uses SALM architecture that cannot be saved as .nemo.")
    print("  It will download automatically from HuggingFace on first use (~5GB)")
    print("  and be cached for offline use after that.")
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
        print("  • Canary:   Downloads from HuggingFace on first use (~5GB)")
        print("              (then cached for offline use)")
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

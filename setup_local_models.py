#!/usr/bin/env python3
"""
Multi-Model Setup Script: Downloads and saves all Parakeet & Canary models as .nemo files.

This script supports downloading all 4 ASR models with unique filenames:
  - Parakeet-TDT-0.6B-v3: parakeet-0.6b-v3.nemo (~2.4 GB)
  - Parakeet-TDT-1.1B:    parakeet-1.1b.nemo    (~4.5 GB)
  - Canary-1B:            canary-1b.nemo        (~5.0 GB)
  - Canary-1B-v2:         canary-1b-v2.nemo     (~5.0 GB)

Features:
  - Menu-driven model selection
  - Download one or multiple models
  - Batch download all models
  - Check what's already downloaded
  - Verify file integrity after download

See docs/manual/comprehensive-asr-model-integration-guide.md for details.
"""

import nemo.collections.asr as nemo_asr
import os
import sys

# ============================================================================
# MODEL DEFINITIONS
# ============================================================================
# Each model has a unique filename to prevent overwrites

MODELS_TO_DOWNLOAD = {
    "1": {
        "model_id": "nvidia/parakeet-tdt-0.6b-v3",
        "filename": "parakeet-0.6b-v3.nemo",
        "display_name": "Parakeet-TDT-0.6B-v3 (Multilingual)",
        "download_size": "~1.2 GB",
        "saved_size": "~2.4 GB",
        "min_size_gb": 1.5,  # Minimum expected file size in GB for validation
        "description": "25 languages, auto-detection, 3,380× RTFx",
        "recommended": True
    },
    "2": {
        "model_id": "nvidia/parakeet-tdt-1.1b",
        "filename": "parakeet-1.1b.nemo",
        "display_name": "Parakeet-TDT-1.1B (Maximum Accuracy)",
        "download_size": "~2.2 GB",
        "saved_size": "~4.5 GB",
        "min_size_gb": 3.0,
        "description": "English only, 1.5% WER (best accuracy)",
        "recommended": False
    },
    "3": {
        "model_id": "nvidia/canary-1b",
        "filename": "canary-1b.nemo",
        "display_name": "Canary-1B (Multilingual + Translation)",
        "download_size": "~2.5 GB",
        "saved_size": "~5.0 GB",
        "min_size_gb": 3.5,
        "description": "25 languages, speech-to-text translation",
        "recommended": False
    },
    "4": {
        "model_id": "nvidia/canary-1b-v2",
        "filename": "canary-1b-v2.nemo",
        "display_name": "Canary-1B v2 (Multilingual + Translation)",
        "download_size": "~2.5 GB",
        "saved_size": "~5.0 GB",
        "min_size_gb": 3.5,
        "description": "25 languages, improved speech translation",
        "recommended": False
    }
}


def create_local_models_directory():
    """Create local_models directory if it doesn't exist."""
    if not os.path.exists("local_models"):
        os.makedirs("local_models")
        print("✓ Created local_models/ directory")
    else:
        print("✓ local_models/ directory already exists")


def get_model_status():
    """Check which models are already downloaded.
    
    Returns:
        dict: Model choice -> status info (exists, size, valid)
    """
    status = {}
    for choice, model in MODELS_TO_DOWNLOAD.items():
        filepath = os.path.join("local_models", model["filename"])
        if os.path.exists(filepath):
            size_gb = os.path.getsize(filepath) / (1024**3)
            is_valid = size_gb >= model["min_size_gb"]
            status[choice] = {
                "exists": True,
                "size_gb": size_gb,
                "valid": is_valid,
                "path": filepath
            }
        else:
            status[choice] = {
                "exists": False,
                "size_gb": 0,
                "valid": False,
                "path": filepath
            }
    return status


def display_model_status():
    """Display current status of all models."""
    print("\n" + "="*80)
    print("📊 CURRENT MODEL STATUS")
    print("="*80)
    
    status = get_model_status()
    total_size = 0
    downloaded_count = 0
    
    for choice, model in MODELS_TO_DOWNLOAD.items():
        s = status[choice]
        if s["exists"]:
            downloaded_count += 1
            total_size += s["size_gb"]
            if s["valid"]:
                print(f"\n  {choice}. {model['display_name']}")
                print(f"     ✅ Downloaded ({s['size_gb']:.2f} GB)")
                print(f"     📁 {s['path']}")
            else:
                print(f"\n  {choice}. {model['display_name']}")
                print(f"     ⚠️  Downloaded but may be corrupted ({s['size_gb']:.2f} GB)")
                print(f"        Expected minimum: {model['min_size_gb']} GB")
        else:
            print(f"\n  {choice}. {model['display_name']}")
            print(f"     ❌ Not downloaded")
            print(f"        Size: {model['download_size']} → {model['saved_size']}")
    
    print(f"\n" + "-"*40)
    print(f"📦 Total: {downloaded_count}/{len(MODELS_TO_DOWNLOAD)} models ({total_size:.2f} GB)")
    print("="*80 + "\n")


def download_and_save_model(choice):
    """Download and save a single model as .nemo file.
    
    Args:
        choice: Model choice key from MODELS_TO_DOWNLOAD
        
    Returns:
        bool: True if successful, False otherwise
    """
    if choice not in MODELS_TO_DOWNLOAD:
        print(f"❌ Invalid choice: {choice}")
        return False
    
    model = MODELS_TO_DOWNLOAD[choice]
    output_path = os.path.abspath(f"local_models/{model['filename']}")
    
    print("\n" + "="*80)
    print(f"📦 DOWNLOADING: {model['display_name']}")
    print("="*80)
    print(f"   Model ID: {model['model_id']}")
    print(f"   Output: {output_path}")
    print(f"   Download size: {model['download_size']}")
    print(f"   Saved size: {model['saved_size']}")
    print(f"   Description: {model['description']}")
    print("\nThis may take 2-10 minutes depending on your connection...\n")
    
    try:
        # Download from HuggingFace
        print("   ⏳ Downloading from HuggingFace...")
        asr_model = nemo_asr.models.ASRModel.from_pretrained(model["model_id"])
        print(f"   ✓ Downloaded {model['display_name']}")
        
        # Save to local file
        print(f"   💾 Saving to {model['filename']}...")
        asr_model.save_to(output_path)
        print(f"   ✓ Saved to {output_path}")
        
        # Verify file
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path) / (1024**3)
            print(f"\n   📊 Verification:")
            print(f"      File size: {file_size:.2f} GB")
            
            if file_size < model["min_size_gb"]:
                print(f"      ⚠️  Warning: File smaller than expected (min: {model['min_size_gb']} GB)")
                print(f"         This may indicate a corrupted download.")
                return False
            else:
                print(f"      ✅ File size OK (expected min: {model['min_size_gb']} GB)")
        else:
            print(f"\n   ❌ ERROR: File was not created at {output_path}")
            return False
        
        # Clean up memory
        del asr_model
        
        print(f"\n✅ {model['display_name']} ready for use!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error downloading {model['display_name']}:")
        print(f"   {type(e).__name__}: {str(e)}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        return False


def download_all_models():
    """Download all models in sequence."""
    print("\n" + "="*80)
    print("📦 BATCH DOWNLOAD: ALL MODELS")
    print("="*80)
    print("\nThis will download all 4 models.")
    print("Estimated total size: ~16-17 GB")
    print("Estimated time: 15-45 minutes (depends on connection)")
    
    confirm = input("\nProceed with batch download? (y/n): ").strip().lower()
    if confirm not in ('y', 'yes'):
        print("\n❌ Batch download cancelled")
        return
    
    results = {}
    for choice in MODELS_TO_DOWNLOAD.keys():
        success = download_and_save_model(choice)
        results[choice] = success
    
    # Summary
    print("\n" + "="*80)
    print("📊 BATCH DOWNLOAD SUMMARY")
    print("="*80)
    
    success_count = sum(1 for s in results.values() if s)
    for choice, success in results.items():
        model = MODELS_TO_DOWNLOAD[choice]
        status = "✅ Success" if success else "❌ Failed"
        print(f"   {model['display_name']}: {status}")
    
    print(f"\n   Total: {success_count}/{len(MODELS_TO_DOWNLOAD)} models downloaded successfully")
    print("="*80 + "\n")


def display_menu():
    """Display the main menu and get user selection."""
    print("\n" + "="*80)
    print("📦 NeMo ASR Local Model Setup")
    print("="*80)
    print("\nSelect an option:\n")
    
    for choice, model in MODELS_TO_DOWNLOAD.items():
        rec = " [RECOMMENDED]" if model.get("recommended") else ""
        print(f"  {choice}. Download {model['display_name']}{rec}")
        print(f"     Size: {model['download_size']} → {model['saved_size']}")
        print(f"     {model['description']}\n")
    
    print("  5. Download ALL models (batch mode)")
    print("  6. Check what's already downloaded")
    print("  0. Exit\n")
    
    return input("Enter your choice: ").strip()


def main():
    print("\n" + "="*80)
    print("🚀 NeMo ASR Multi-Model Setup Script")
    print("="*80)
    print("\nThis script downloads ASR models and saves them as local .nemo files.")
    print("Local files load faster and work completely offline.")
    print("\nFeatures:")
    print("  • Download individual models or all at once")
    print("  • Unique filenames prevent overwrites")
    print("  • Automatic integrity verification")
    print("  • Works with transcribe_ui.py fallback loading")
    
    # Create directory
    create_local_models_directory()
    
    while True:
        choice = display_menu()
        
        if choice == "0":
            print("\n👋 Goodbye!")
            break
        elif choice == "5":
            download_all_models()
        elif choice == "6":
            display_model_status()
        elif choice in MODELS_TO_DOWNLOAD:
            success = download_and_save_model(choice)
            if success:
                print("\n✅ Download complete!")
            else:
                print("\n❌ Download failed. See error above.")
        else:
            print("\n❌ Invalid choice. Please try again.")
        
        input("\nPress Enter to continue...")


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

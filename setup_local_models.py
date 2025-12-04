#!/usr/bin/env python3
"""
Simplified setup: Only downloads and saves Parakeet as .nemo file.
Canary will load from HuggingFace cache automatically.
"""

import nemo.collections.asr as nemo_asr
import os
import sys

def setup_parakeet():
    """Download Parakeet and save as .nemo file."""
    
    print("\\n" + "="*80)
    print("📦 Setting Up Parakeet-TDT-0.6B v2")
    print("="*80)
    print("\\nThis will:")
    print("  1. Download Parakeet from HuggingFace (~1.2 GB)")
    print("  2. Save as local .nemo file")
    print("  3. Take ~5-10 minutes")
    print("\\nNote: Canary will be set up automatically by GitHub Copilot")
    print("      to load from HuggingFace cache (no .nemo file needed)\\n")
    
    response = input("Continue? (y/n): ").strip().lower()
    if response != 'y':
        print("❌ Setup cancelled")
        sys.exit(0)
    
    try:
        # Create directory
        os.makedirs("local_models", exist_ok=True)
        print("✓ Created local_models/ directory")
        
        # Download Parakeet
        print("\\n📥 Downloading Parakeet...")
        model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v2")
        print("✓ Download complete")
        
        # Save as .nemo
        print("\\n💾 Saving to local_models/parakeet.nemo...")
        output_path = os.path.abspath("local_models/parakeet.nemo")
        model.save_to(output_path)
        
        # Verify
        if os.path.exists(output_path):
            size = os.path.getsize(output_path) / (1024**3)
            print(f"✓ Saved successfully ({size:.2f} GB)")
            
            print("\\n" + "="*80)
            print("✅ Parakeet Setup Complete!")
            print("="*80)
            print(f"\\nSaved to: {output_path}")
            print("\\nNext steps:")
            print("  1. Upload canary-hybrid-loading-fix.md to GitHub")
            print("  2. Let GitHub Copilot implement Canary loading")
            print("  3. Run transcribe_ui.py")
            print("\\nCanary will download on first use (~5GB), then load from cache")
            print("="*80 + "\\n")
            return True
        else:
            print("❌ File was not created")
            return False
            
    except Exception as e:
        print(f"\\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    try:
        if not setup_parakeet():
            print("\\n❌ Setup failed. Please check error above.")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\\n\\n❌ Setup interrupted")
        sys.exit(1)

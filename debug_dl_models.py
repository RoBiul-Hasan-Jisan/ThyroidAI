# ============================================
# debug_dl_models.py - Run this locally
# ============================================
import os

MODEL_DIR = "D:/Aweb/thyroid-ai-platform/backend/models"
DL_MODEL_DIR = os.path.join(MODEL_DIR, "dl_models")

print("="*70)
print("🔍 DEBUGGING DL MODELS")
print("="*70)
print(f"DL Model Directory: {DL_MODEL_DIR}")
print(f"Exists: {os.path.exists(DL_MODEL_DIR)}")

if os.path.exists(DL_MODEL_DIR):
    print(f"\n📂 Files in dl_models/:")
    for file in sorted(os.listdir(DL_MODEL_DIR)):
        file_path = os.path.join(DL_MODEL_DIR, file)
        size = os.path.getsize(file_path) / 1024
        print(f"   📄 {file} ({size:.1f} KB)")
    
    print(f"\n📊 Summary:")
    keras_files = [f for f in os.listdir(DL_MODEL_DIR) if f.endswith('.keras')]
    h5_files = [f for f in os.listdir(DL_MODEL_DIR) if f.endswith('.h5')]
    json_files = [f for f in os.listdir(DL_MODEL_DIR) if f.endswith('.json')]
    
    print(f"   .keras files: {len(keras_files)}")
    print(f"   .h5 files: {len(h5_files)}")
    print(f"   .json files: {len(json_files)}")
    
    print("\n📋 Model names from metadata:")
    import json
    with open(os.path.join(MODEL_DIR, "metadata.json")) as f:
        metadata = json.load(f)
    
    for model in metadata.get("all_models", []):
        if model.get("type") == "DL":
            print(f"   - {model.get('model')}")
else:
    print("❌ dl_models directory not found!")
print("="*70)
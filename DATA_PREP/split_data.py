import os
import shutil
import random
from tqdm import tqdm

INPUT_FOLDER = 'dataset_gabungan_enhanced' 

OUTPUT_FOLDER = 'dataset_final_ready'

TRAIN_RATIO = 0.8
VAL_RATIO   = 0.1
TEST_RATIO  = 0.1 

def split_dataset():
    if not os.path.exists(INPUT_FOLDER):
        print(f"❌ Error: Folder input '{INPUT_FOLDER}' tidak ditemukan!")
        return

    if os.path.exists(OUTPUT_FOLDER):
        print(f"🧹 Membersihkan folder lama '{OUTPUT_FOLDER}'...")
        shutil.rmtree(OUTPUT_FOLDER)

    classes = ['Normal', 'Tuberculosis']

    for class_name in classes:
        print(f"\n📂 Memproses kelas: {class_name}...")
 
        src_path = os.path.join(INPUT_FOLDER, class_name)
        if not os.path.exists(src_path):
            print(f"⚠️ Warning: Folder kelas {class_name} tidak ditemukan, skip.")
            continue

        files = [f for f in os.listdir(src_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        random.shuffle(files)
        
        count = len(files)
        train_count = int(count * TRAIN_RATIO)
        val_count = int(count * VAL_RATIO)
        test_count = count - train_count - val_count
        
        print(f"   Total: {count} gambar")
        print(f"   Train: {train_count} | Val: {val_count} | Test: {test_count}")

        train_files = files[:train_count]
        val_files = files[train_count:train_count+val_count]
        test_files = files[train_count+val_count:]

        def copy_files(file_list, split_type):
            dest_dir = os.path.join(OUTPUT_FOLDER, split_type, class_name)
            os.makedirs(dest_dir, exist_ok=True)
            
            for f in tqdm(file_list, desc=f"   Copying to {split_type}"):
                shutil.copy2(os.path.join(src_path, f), os.path.join(dest_dir, f))

        copy_files(train_files, 'train')
        copy_files(val_files, 'validation')
        copy_files(test_files, 'test')

if __name__ == "__main__":
    print("=== MEMULAI PEMBAGIAN DATASET (SPLIT) ===")
    split_dataset()
    print("\n" + "="*40)
    print("✅ SELESAI! Dataset siap di folder:", OUTPUT_FOLDER)
    print("="*40)
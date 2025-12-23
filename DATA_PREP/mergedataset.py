import os
import shutil
import hashlib
from tqdm import tqdm

OLD_DATASET_DIR = 'dataset_split' 

NEW_DATASET_DIR = 'dataset_baru_raw' 

OUTPUT_DIR = 'dataset_gabungan_raw' 


def calculate_md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def collect_images(base_dir, source_name):
    image_list = []
    categories = ['Normal', 'Tuberculosis']
    
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                folder_name = os.path.basename(root)
                
                label = None
                if folder_name in categories:
                    label = folder_name
                
                if label:
                    full_path = os.path.join(root, file)
                    image_list.append({
                        'path': full_path,
                        'label': label,
                        'source': source_name
                    })
    return image_list

def main():
    if not os.path.exists(NEW_DATASET_DIR):
        print(f"❌ Error: Folder dataset baru '{NEW_DATASET_DIR}' tidak ditemukan!")
        return

    print("🔍 Sedang memindai gambar...")

    old_images = collect_images(OLD_DATASET_DIR, "OLD")
    new_images = collect_images(NEW_DATASET_DIR, "NEW")
    all_images = old_images + new_images
    
    print(f"   Ditemukan {len(old_images)} gambar dari dataset lama.")
    print(f"   Ditemukan {len(new_images)} gambar dari dataset baru.")
    print(f"   Total kandidat: {len(all_images)} gambar.")

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    for cat in ['Normal', 'Tuberculosis']:
        os.makedirs(os.path.join(OUTPUT_DIR, cat), exist_ok=True)

    print("\n🚀 Memulai penggabungan pintar (Cek Duplikat)...")
    
    seen_hashes = set()
    count_saved = 0
    count_duplicate = 0
    
    for img_info in tqdm(all_images):
        src_path = img_info['path']
        label = img_info['label']
        
        file_hash = calculate_md5(src_path)
        
        if file_hash in seen_hashes:
            count_duplicate += 1
            continue
        
        seen_hashes.add(file_hash)
        
        ext = os.path.splitext(src_path)[1]
        new_filename = f"merged_{label}_{count_saved+1:05d}{ext}"
        dst_path = os.path.join(OUTPUT_DIR, label, new_filename)

        shutil.copy2(src_path, dst_path)
        count_saved += 1

    print("\n" + "="*40)
    print("✅ PENGGABUNGAN SELESAI!")
    print(f"📂 Folder Hasil: {OUTPUT_DIR}")
    print(f"📥 Total Gambar Disimpan: {count_saved}")
    print(f"🗑️  Duplikat Dibuang: {count_duplicate}")
    print("="*40)

if __name__ == "__main__":
    main()
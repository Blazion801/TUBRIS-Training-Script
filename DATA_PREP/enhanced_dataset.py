import cv2
import os
import numpy as np
from tqdm import tqdm

INPUT_DIR = 'dataset_gabungan_raw'   

OUTPUT_DIR = 'dataset_gabungan_enhanced' 

def apply_clahe(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"Warning: Gagal membaca {image_path}")
        return None
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced_img = clahe.apply(img)

    enhanced_img_rgb = cv2.cvtColor(enhanced_img, cv2.COLOR_GRAY2RGB)
    
    return enhanced_img_rgb

def process_dataset():

    categories = ['Normal', 'Tuberculosis']
    
    if not os.path.exists(INPUT_DIR):
        print(f"❌ Error: Folder input '{INPUT_DIR}' tidak ditemukan!")
        return

    for label in categories:
        input_path = os.path.join(INPUT_DIR, label)
        output_path = os.path.join(OUTPUT_DIR, label)
        
        if not os.path.exists(input_path):
            print(f"⚠️ Warning: Folder {input_path} tidak ditemukan, skip.")
            continue
            
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        print(f"🚀 Sedang memproses kategori: {label}...")
        
        files = os.listdir(input_path)
        
        for filename in tqdm(files):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_src = os.path.join(input_path, filename)
                img_dst = os.path.join(output_path, filename)
                
                enhanced_img = apply_clahe(img_src)
                
                if enhanced_img is not None:
                    cv2.imwrite(img_dst, enhanced_img)

if __name__ == "__main__":
    print("=== MEMULAI PENAJAMAN CITRA (CLAHE) ===")
    print(f"Input: {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print("-" * 30)
    
    process_dataset()
    
    print("\n" + "="*40)
    print("✅ SELESAI! Dataset tajam siap digunakan.")
    print(f"📂 Lokasi: {OUTPUT_DIR}")
    print("="*40)
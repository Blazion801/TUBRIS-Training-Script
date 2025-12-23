import cv2
import matplotlib.pyplot as plt
import os

NORMAL_IMG = r'D:\PROJECT\TUBRIS AI MODEL\TB_Chest_Radiography_Database\Normal\Normal-10.png' 

TB_IMG = r'D:\PROJECT\TUBRIS AI MODEL\TB_Chest_Radiography_Database\Tuberculosis\Tuberculosis-10.png' 

def process_and_show(img_path, title_prefix):
    if not os.path.exists(img_path):
        print(f"❌ Error: Gambar tidak ditemukan di {img_path}")
        return None, None

    img_original = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_clahe = clahe.apply(img_original)
    
    return img_original, img_clahe

norm_orig, norm_clahe = process_and_show(NORMAL_IMG, "Normal")
tb_orig, tb_clahe = process_and_show(TB_IMG, "Tuberculosis")

if norm_orig is not None and tb_orig is not None:
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.imshow(norm_orig, cmap='gray')
    plt.title("Normal - ASLI (Mentah)")
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(norm_clahe, cmap='gray')
    plt.title("Normal - CLAHE (Enhanced)")
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(tb_orig, cmap='gray')
    plt.title("TBC - ASLI (Mentah)")
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(tb_clahe, cmap='gray')
    plt.title("TBC - CLAHE (Enhanced)")
    plt.axis('off')

    plt.tight_layout()
    plt.show()
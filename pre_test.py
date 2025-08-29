
import os
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image

input_dir = "/home/lcj/data1_link/TestSetB"
out_dirs = "/home/lcj/data1_link/TestSetB_CLAHE"

os.makedirs(out_dirs, exist_ok=True)
   

for img_name in tqdm(sorted(os.listdir(input_dir))):
    _, ext = os.path.splitext(img_name)

    img_path = os.path.join(input_dir, img_name)

    data = None

    data = np.fromfile(img_path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)  # 读取为 BGR, HxWx3

    if img is None:
        raise FileNotFoundError(f"Failed to load image: {img_path}. "
                                "Check that the file exists, is readable and is a valid image file.")

    # best 0.8205 
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=0.4, tileGridSize=(8,8))
    cl = clahe.apply(l)
    lab_clahe = cv2.merge((cl, a, b))
    clahe_bgr = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    out_path_clahe = os.path.join(out_dirs, img_name)
    _, buf_clahe = cv2.imencode(ext, clahe_bgr)
    buf_clahe.tofile(out_path_clahe)


print("Done.")

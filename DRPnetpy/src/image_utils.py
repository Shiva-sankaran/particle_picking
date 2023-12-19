import cv2
import numpy as np
import mrcfile
import matplotlib.pyplot as plt
import os

def SetScaleFactors(inpath, box_size):
    IM_SIZE_REF = 1800
    r_factor = 1.5
    rbox_ref = 32
    rbox_large = 96

    flist = [f for f in os.listdir(inpath) if f.endswith('.mrc')]
    fname = flist[0]
    with mrcfile.open(os.path.join(inpath, fname), permissive=True) as mrc:
        map0 = mrc.data
        dims = map0.shape

    # box_size: input particle size (in pixels)
    rbox = box_size / 2

    if rbox > rbox_large * r_factor:
        rbox_ref = rbox_ref * r_factor

    if rbox / rbox_ref > 2:
        f1 = np.ceil(rbox / rbox_ref)
    elif rbox / rbox_ref > 1:
        f1 = 2
    else:
        f1 = 1

    f2 = 1
    if dims[0] / f1 > IM_SIZE_REF:
        f2 = np.ceil(dims[0] / IM_SIZE_REF)

    scale_factor = f1 * f2
    rbox_scale = np.floor(rbox / scale_factor)

    if rbox_scale < rbox_ref:
        f3 = 1
        sigma_gauss = 3
    else:
        f3 = 2
        sigma_gauss = 4

    return scale_factor, int(rbox_scale), int(sigma_gauss), int(f3)

def PreprocessImage(image_path, resize_factor, is_negative, is_train):
    global global_img2
    # Read micrograph mrc format
    img_0 = mrcfile.open(image_path, permissive=True).data
    # Normalize
    if np.max(img_0) > 255 * 255:
        mi = np.min(img_0)
        ma = np.max(img_0)
        img_0 = (img_0 - mi) / (ma - mi)
    elif np.max(img_0) > 255:
        img_0 = img_0 / (255 * 255)
    else:
        img_0 = img_0 / 255.0

    # Resize, rotate, and smooth filter
    img_1 = cv2.resize(img_0, (0, 0), fx=1/resize_factor, fy=1/resize_factor,interpolation=cv2.INTER_CUBIC)
    # img_1 = np.rot90(img_1)
    # img_1 = np.flipud(img_1)
    fsize = 0.5 if is_train == 1 else 2
    ksize = (int(2 * np.ceil(2 * fsize) + 1),int(2 * np.ceil(2 * fsize) + 1))
    img_2 = cv2.GaussianBlur(img_1, ksize, sigmaX=fsize)
    # Stretch intensity range
    lo,hi = np.percentile(img_2, (1, 99))
    img_3 = np.clip((img_2 - lo) / (hi - lo), 0, 1)
    mi = np.mean(img_3)
    if mi < 0.5 or mi > 0.6:
        dist = 0.55 - mi
        img_3b = np.round(img_3 * 255) + np.round(dist * 255)
        img_3b = np.clip(img_3b, 0, 255).astype(np.uint8)
        img_3c = img_3b.astype(np.float32) / 255.0
        img_3 = img_3c
    
    # Illumination correction
    ksize_ll = (int(2 * np.ceil(2 * 100) + 1),int(2 * np.ceil(2 * 100) + 1))
    lpf = cv2.GaussianBlur(img_3, ksize_ll, 100)
    meanLpf = np.mean(lpf)
    img_4 = img_3 - lpf + meanLpf
    clean_image = img_4
    
    # Switch to negative stain
    if is_negative == 1:
        clean_image = cv2.bitwise_not(clean_image)
    
    return clean_image


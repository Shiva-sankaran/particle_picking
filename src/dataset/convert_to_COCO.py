import os
import glob
import numpy as np
from PIL import Image
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
import shutil
from tqdm import tqdm


DATA_DIR = "/home/shiva/projects/particle_picking/data/cryopp/cryopp_lite"
YOLO_DIR = "/home/shiva/projects/particle_picking/data/cryopp/cryoppp_lite_YOLO"
COCO_IMAGE_DIR = YOLO_DIR + "/images/"
COCO_LABELS_DIR = YOLO_DIR + "/labels/"



def save_annos_yolo_format(codf,image_path,output_file_path=None):
    im = Image.open(image_path)
    w,h = im.size
    temp_df = codf.copy()
    temp_df["Height"] = temp_df["Diameter"]
    temp_df = temp_df.rename(columns={"Diameter":"Width"})
    temp_df["X-Coordinate"] = temp_df["X-Coordinate"]/w
    temp_df["Y-Coordinate"] = temp_df["Y-Coordinate"]/h
    temp_df["Width"] = temp_df["Width"]/w
    temp_df["Height"] = temp_df["Height"]/h

    lines = []
    for item in temp_df.values:
        lines.append("{} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(int(item[2]),item[0],item[1],item[3],item[4]))
    with open(output_file_path,"w") as f:
        f.writelines(lines)
    
    return temp_df


for empiar_dir in tqdm(glob.glob(DATA_DIR + "/*/")):
    empiar_id = empiar_dir.split("/")[-2]
    if((empiar_id == "10671") or (empiar_id == "10389")):
        print("skipping 10671 or 10389")
        continue
    particle_classes_csv_path = empiar_dir+"ground_truth/empiar-{}_particles_selected.csv".format(empiar_id)
    cls_df = pd.read_csv(particle_classes_csv_path)

    total_files = glob.glob(empiar_dir+"ground_truth/particle_coordinates/*.csv")
    train_ratio = 0.8
    validation_ratio = 0.1
    test_ratio = 0.10
    x_train, x_test = train_test_split(total_files, test_size=1 - train_ratio)
    x_val, x_test= train_test_split(x_test, test_size=test_ratio/(test_ratio + validation_ratio)) 
    skipping = 0
    d=  {"train":x_train,"val":x_val,"test":x_test}
    print("Processing {} paths".format(len(total_files)))
    for split,coords_paths in d.items():
        for coords_path in coords_paths:
            particle_file_name = coords_path.split("/")[-1][:-4]
            particle_cls_df = cls_df[cls_df["Particles Filename"] ==particle_file_name + "_particles.mrc"][["X-Coordinate","Y-Coordinate","Class Number"]]
            image_path = empiar_dir + "micrographs/"+particle_file_name + '.jpg'
            if not os.path.isfile(image_path):
                skipping +=1
                continue
            df = pd.read_csv(coords_path)[["X-Coordinate","Y-Coordinate","Diameter"]]
            merged_df = pd.merge(particle_cls_df,df,how="inner",on=["X-Coordinate","Y-Coordinate"])
            anno_file_path = COCO_LABELS_DIR + particle_file_name + ".txt"
            if(split == "test"):
                os.makedirs(COCO_IMAGE_DIR + split +"/" + empiar_id + "/",exist_ok=True)
                shutil.copy(image_path, COCO_IMAGE_DIR + split +"/" + empiar_id + "/")
            else:
                shutil.copy(image_path, COCO_IMAGE_DIR + split +"/")
            save_annos_yolo_format(merged_df,image_path,anno_file_path)


    print("Skipped {} images".format(skipping))

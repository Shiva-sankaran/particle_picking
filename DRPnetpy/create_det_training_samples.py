from image_utils import SetScaleFactors,PreprocessImage
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np
from tqdm import tqdm
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--data_folder_path",type=str,help="Should contain micrographs/ and ground_truth/particle_coordinates/")
parser.add_argument("--output_folder",type=str,help="Should contain np_training_data/ and patches/GT and patches/ORIG")
parser.add_argument("--box_size",type=int,help="Estimated box size (find from config file for this ID)")
parser.add_argument("--is_negative_stain",type=bool, default= False,help="Is data negative stained")
parser.add_argument("--is_train",type=int,help="Is data for train (0/1)")
args = parser.parse_args()

def cropsquare_up(I, X, Y, R):

    channels = I.shape[2]
    X1 = X - R
    X2 = X + R
    Y1 = Y - R
    Y2 = Y + R
    w = 2 * R

    indexX = np.where(X1 <= 0)
    indexY = np.where(Y1 <= 0)
    imgwidth = I.shape[0]
    indeX2 = X2 > imgwidth
    indeY2 = Y2 > imgwidth
    X2[indeX2] = imgwidth
    Y2[indeY2] = imgwidth
    X1[indexX] = 1
    Y1[indexY] = 1

    count = len(X1)
    IC = np.zeros((w, w, channels, count))

    for i in range(count):
        img = I[X1[i]:X2[i], Y1[i]:Y2[i], :]
        x, y, _ = img.shape
        if x != w or y != w:
            img = cv2.resize(img, (w, w))
            img = np.expand_dims(img,axis=-1)
        IC[:, :, :, i] = img

    return IC
def CreateDetTrainingSamples(folder_path,box_size,output_folder,is_negative_stain=False,is_train=0):
    mrc_folder_path = os.path.join(folder_path,"micrographs/")
    particle_coord_folder_path = os.path.join(folder_path,"ground_truth/particle_coordinates/")
    scale_factor, rbox_scale, sigma_gauss, f3 = SetScaleFactors(mrc_folder_path,box_size)
    mrc_file_paths = list(glob.glob(mrc_folder_path + "*mrc"))
    r_patch = rbox_scale
    patches = []
    images_data = []
    images_labels = []
    total_skips = 0
    actual_present = 0

    for mrc_file_path in tqdm(mrc_file_paths):
        print(mrc_file_path)
        fname = mrc_file_path.strip().split("/")[-1]
        cname = particle_coord_folder_path + fname[:-4] + ".csv"
        if( not os.path.exists(cname)):
            print("No coord file for ",fname)
            total_skips+=1
            continue
        actual_present+=1
        anno_df = pd.read_csv(cname)
        im = PreprocessImage(mrc_file_path,scale_factor,is_negative_stain,is_train)
        # plt.figure(figsize=(10,10))
        # plt.imshow(im,cmap="gray")
        # plt.savefig("/home/shiva/projects/particle_picking/DRPnet-py/temp_data/ppo_main_fun.svg",dpi=300)
        dims = im.shape
        w,h = im.shape
        coords = anno_df[["X-Coordinate","Y-Coordinate"]].values
        coords = coords/scale_factor
        for i in range(len(coords)):
            coords[i][1] = h - coords[i][1] 

        mask_centers = np.zeros((dims[0], dims[1]), dtype=np.uint8)
        for coord in coords:
            mask_centers[int(coord[0]), int(coord[1])] = 1

        centers = np.argwhere(mask_centers == 1)
        for c in centers:
            cv2.circle(mask_centers, (int(c[0]), int(c[1])), radius=int((rbox_scale // 3))+2, color=255, thickness=-1)

        inverted_mask = np.logical_not(mask_centers.astype(bool)).astype(np.uint8)
        d_center = cv2.distanceTransform(inverted_mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        cells = d_center < (rbox_scale / 3.5)
        d_border = cv2.distanceTransform(cells.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        K = np.square(d_border)
        
        K = np.expand_dims(K,axis = -1)
        im = np.expand_dims(im,axis = -1)
        xy = centers

        
        GT_NEW = cropsquare_up(K, xy[:, 1], xy[:, 0], r_patch)
        original_image = cropsquare_up(im, xy[:, 1], xy[:, 0], r_patch)
        


        num_patches = GT_NEW.shape[3]
        num_prev = len(patches)

        patches_folder = output_folder + "/patches/"

        for k in range(num_patches):
            image_name1 = f'{patches_folder}/GT/{fname[:-4]}_{k}.png'
            image_name2 = f'{patches_folder}/ORIG/{fname[:-4]}_{k}.png'

            cv2.imwrite(image_name1,GT_NEW[:,:,:,k])
            cv2.imwrite(image_name2,original_image[:,:,:,k]*255)

            patches.append({'names': image_name2})

        images_data.append(original_image)
        images_labels.append(GT_NEW)


    print("Total mrc skips due to coord file not being present: ",total_skips)
    print("Actual present: ",actual_present)

    image_data = np.concatenate(images_data,axis=3)
    images_labels = np.concatenate(images_labels,axis=3)
    print(image_data.shape)
    print(images_labels.shape)
    np_save_path = output_folder + "/np_training_data/"

    np.save(np_save_path + "orig_patch_data.npy",image_data)

    np.save(np_save_path + "labels_patch_data.npy",images_labels)
    print("SAVED")


# print(args.is_negative_stain)
# folder_path = "/home/shiva/projects/particle_picking/data/cryoppp_full/10017/"
# box_size = 192
# ouput_folder = "/home/shiva/projects/particle_picking/DRPnet-py/data/10005"
# CreateDetTrainingSamples(folder_path,box_size,ouput_folder,is_negative_stain=False,is_train=1)



# folder_path = "/home/shiva/projects/particle_picking/data/cryoppp_full/10005/"
# box_size = 192
# ouput_folder = "/home/shiva/projects/particle_picking/DRPnet-py/data/10005"

CreateDetTrainingSamples(args.data_folder_path,args.box_size,args.output_folder,args.is_negative_stain,args.is_train)

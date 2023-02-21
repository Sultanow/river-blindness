import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import json

from enum import Enum
from unetModules.unet_model import UNet
from unetModules.Utils import Utils
from PIL import Image

class VisualOption(Enum):
    contour = "contour"
    alpha = "alpha"
    paintBackground = "paintBackground"

    def __str__(self):
        return self.value

class dataset(Enum):
    riverblindness = "riverblindness"
    schistosoma = "schistosoma"

    def __str__(self):
        return self.value

color_map = {
    'background'        : [ 0, 0, 0],
    'segmentedObject'    : [ 255, 255, 255]
}

# Data structure for contour polygons
class Polygon:
    points = []
    area = 0
    
    def __init__(self, polygon_points):
        self.points = polygon_points
        
        x_coordinates = [point[0][0] for point in self.points]
        y_coordiantes = [point[0][1] for point in self.points]
              
        # calculation polygon area using shoelace algorithm
        self.area = 0.5 * np.abs(np.dot(x_coordinates,np.roll(y_coordiantes,1))-np.dot(y_coordiantes,np.roll(x_coordinates, 1)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-ds", "--dataset",
                    type=dataset,
                    choices=list(dataset),
                    help="Which model for a specific dataset. riverblindness or schistosoma")

    parser.add_argument("-i", "--imagePath",
                type=str,
                help="The path to the image to perform semantic segmentation")

    
    parser.add_argument("-vo",
                type=VisualOption,
                choices=list(VisualOption),
                help="Selecting a visualisation option for inference")

    parser.add_argument("-a", "--alphaValue",
                type=float,
                default=0.0,
                help="Alpha value to use for alhpa visualisation option.")
    
    parser.add_argument("-r",
            action="store_true",
            help="Saving inference as result image. If False a plot of stepwise images will be saved.")

    parser.add_argument("-at", "--areaThreshold",
            type=int,
            default=30000,
            help="Area Threshold for primary Worm/Egg")


scriptParameter, unparesed = parser.parse_known_args()
device = "cuda:0"

if(scriptParameter.dataset == dataset.riverblindness):
    #ckpt-unet-65-190.5087480507791.pth
    print("[INFO] Using riverblindes model.")
    trained_model = torch.load("./modelBackups/RiverBlindnessEvaluation_13_02_23_best/ckpt-unet-65-190.5087480507791.pth",  map_location=device)
elif(scriptParameter.dataset == dataset.schistosoma):
    print("[INFO] Using schistosoma model.")
    trained_model = torch.load("./modelBackups/SchistosomaMansoni_Unet_10_02_2023_best/ckpt-unet-100-20.14962317608297.pth",  map_location=device)

enet = UNet(3, 2)
enet.load_state_dict(trained_model['state_dict'])
enet = enet.to(device)

print("[INFO] Loaded image.")
input_img_to_predict = Image.open(scriptParameter.imagePath)
input_img_to_predict = cv2.resize(np.array(input_img_to_predict), (512, 512), cv2.INTER_NEAREST)
input_img_tensor = torch.tensor(input_img_to_predict[:,:,0:3]).unsqueeze(0).transpose(2, 3).transpose(1, 2)
input_img_tensor = input_img_tensor.to(device)

print("[INFO] Predicting image.")
with torch.no_grad():   
    out = enet(input_img_tensor.float()).squeeze(0)
    
segmentedImg = out.data.max(0)[1].cpu().numpy()
segmentedImg = segmentedImg.astype('uint8')

print("[INFO] Use erosion and dilatation.")
erosion_dest = cv2.erode(segmentedImg, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (40, 40)))
dilation_dest = cv2.dilate(erosion_dest, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50)))

segmented_img_after_dil_ero = dilation_dest

contours, hierarchy = cv2.findContours(segmented_img_after_dil_ero, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

segmented_polygons = []

for polygon_pionts in contours:
    segmented_polygons.append(Polygon(polygon_pionts))

primary_worm_polygons_contours = []

print("[INFO] Applying area treshold.")
for polygon in segmented_polygons:
    if(polygon.area > scriptParameter.areaThreshold):
        primary_worm_polygons_contours.append(polygon.points)

input_img_copy_to_save = np.copy(input_img_to_predict)

if(scriptParameter.vo == VisualOption.contour):
    print("[INFO] Using kontour visualisation option.")
    input_img_copy_to_save = cv2.drawContours(input_img_copy_to_save, primary_worm_polygons_contours, -1, (255, 0, 0), 3)

elif(scriptParameter.vo == VisualOption.alpha):
    print("[INFO] Using alpha visualisation option.")

    if(input_img_copy_to_save.shape[2] < 4):
        input_img_copy_to_save = cv2.cvtColor(np.array(input_img_copy_to_save), cv2.COLOR_BGR2BGRA)

    for iy, ix in np.ndindex(input_img_copy_to_save.shape[:2]):
        input_img_copy_to_save[iy, ix, 3] = 255 * scriptParameter.alphaValue

    for contour in primary_worm_polygons_contours:
        for iy, ix in np.ndindex(input_img_copy_to_save.shape[:2]):
            if(cv2.pointPolygonTest(contour, (ix, iy), False) == 1):
                input_img_copy_to_save[iy, ix, 3] = 255

elif(scriptParameter.vo == VisualOption.paintBackground):
    print("[INFO] Using paintBackground visualisation option.")
    img_copy_for_primary_worms = np.copy(input_img_to_predict)

    for iy, ix in np.ndindex(input_img_copy_to_save.shape[:2]):
        input_img_copy_to_save[iy, ix, 0] = 0
        input_img_copy_to_save[iy, ix, 1] = 0
        input_img_copy_to_save[iy, ix, 2] = 0

    for contour in primary_worm_polygons_contours:
        for iy, ix in np.ndindex(input_img_copy_to_save.shape[:2]):
            if(cv2.pointPolygonTest(contour, (ix, iy), False) == 1):
                input_img_copy_to_save[iy, ix, 0] = img_copy_for_primary_worms[iy, ix, 0]
                input_img_copy_to_save[iy, ix, 1] = img_copy_for_primary_worms[iy, ix, 1]
                input_img_copy_to_save[iy, ix, 2] = img_copy_for_primary_worms[iy, ix, 2]

if(scriptParameter.r):
    print("[INFO] Saving Image.")
    output_image = Image.fromarray(input_img_copy_to_save)
    output_image.save("./segmentedImages/segmentedImage.png")
else:
    print("[INFO] Saving Plot.")

    fig, axs = plt.subplots(1, 3, figsize=(40, 40))
    axs[0].imshow(input_img_to_predict)
    axs[1].imshow(Utils.decode_segmap(segmentedImg, color_map))
    axs[2].imshow(input_img_copy_to_save)

    fig.savefig("./segmentedImages/subplotSegmentation.png")

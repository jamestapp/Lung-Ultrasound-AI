import os
from PIL import Image

os.chdir("T:\\MB_lung_forStudents\\segmentation_approved")

for patientName in os.listdir("masks"):
    for videoName in os.listdir("masks\\" + patientName):
        for imageName in os.listdir("masks\\" + patientName + "\\" + videoName):
            mask_img = Image.open("masks\\" + patientName + "\\" + videoName + "\\" + imageName)
            maskSize = mask_img.size
            img = Image.open("images\\" + patientName + "\\" + videoName + "\\" + imageName)
            imgSize = img.size

            if imgSize != maskSize:
                print(f"Patient: {patientName}, Video: {videoName}, Image: {imageName}")
                print(f"Image Size: {imgSize}, Mask Size: {maskSize}")
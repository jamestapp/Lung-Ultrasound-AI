import os
from PIL import Image

os.chdir("T:\\MB_lung_forStudents\\segmentation_approved\\images")
imgSizes = dict()

for patientName in os.listdir("."):
    for videoName in os.listdir(patientName):
        for imageName in os.listdir(patientName + "\\" + videoName):
            img = Image.open(patientName + "\\" + videoName + "\\" + imageName)
            if img.size in imgSizes.keys():
                imgSizes[img.size] += 1
            else:
                imgSizes[img.size] = 1

print(imgSizes)

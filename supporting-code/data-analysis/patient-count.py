import os
from PIL import Image

''' 
    This script looks through all the masks in the lung segmentation dataset, and adds each colour code to a set 
    This set is then printed at the end to aid in data analysis
'''

os.chdir("T:\\MB_lung_forStudents\\segmentation_approved\\masks")

number_of_patients = len(os.listdir("."))
number_of_videos = sum([len(os.listdir(patientName)) for patientName in os.listdir(".")])
number_of_images = sum(
    [
        len(os.listdir(patientName + "\\" + videoName))
        for patientName in os.listdir(".")
        for videoName in os.listdir(patientName)
     ]
)

print("Number of patients: " + str(number_of_patients))
print("Number of videos: " + str(number_of_videos))
print("Number of images: " + str(number_of_images))


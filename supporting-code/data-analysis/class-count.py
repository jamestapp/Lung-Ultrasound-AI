import os
from PIL import Image

''' 
    This script looks through all the masks in the lung segmentation dataset, and adds each colour code to a set 
    This set is then printed at the end to aid in data analysis
'''

os.chdir("T:\\MB_lung_forStudents\\segmentation_approved\\masks")

label_colours = {
    "Ribs": {(255, 0, 0), (255, 0, 1), (255, 1, 0), (255, 1, 1)},
    "Pleura": {(255, 0, 255), (255, 1, 255)},
    "A-Lines": {(0, 255, 255)},
    "Confluent B-Lines": {(255, 100, 0), (255, 101, 1), (255, 100, 1)},
    "Consolidations": {(0, 255, 0), (1, 255, 0), (1, 255, 1), (150, 0, 75)},
    "Effusions": {(0, 0, 255), (0, 1, 255), (100, 50, 0), (100, 51, 1), (101, 51, 0)}
}

patients = {
    "sample_patient": {
        "classes": {
            "Ribs": 0,
            "Pleura": 0,
            "A-Lines": 0,
            "Confluent B-Lines": 0,
            "Consolidations": 0,
            "Effusions": 0
        },
        "imageCount": 0,
        "videoCount": 0
    }
}

for patientName in os.listdir("."):
    patients[patientName] = {"classes": {}, "imageCount": 0, "videoCount": 0}
    for videoName in os.listdir(patientName):
        patients[patientName["videoCount"]] += 1
        for imageName in os.listdir(patientName + "\\" + videoName):
            patients[patientName["imageCount"]] += 1
            colimg = Image.open(patientName + "\\" + videoName + "\\" + imageName)
            for label in label_colours.keys():
                image_colour_set = {colour[1] for colour in Image.Image.getcolors(colimg)}
                if label_colours[label].intersection(image_colour_set):
                    patients[patientName][label] += 1

print(label_counts_counts)

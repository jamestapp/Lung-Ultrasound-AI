import os
from PIL import Image

''' 
    This script looks through all the masks in the lung segmentation dataset, and adds each colour code to a set 
    This set is then printed at the end to aid in data analysis
'''

os.chdir("T:\\MB_lung_forStudents\\segmentation_approved\\masks")

colours = dict()
graycolours = dict()
for patientName in os.listdir("."):
    #print(os.listdir(patientName))
    for videoName in os.listdir(patientName):
        #print(os.listdir(patientName + "\\" + videoName))
        for imageName in os.listdir(patientName + "\\" + videoName):
            colimg = Image.open(patientName + "\\" + videoName + "\\" + imageName)
            for colour in {colour[1] for colour in Image.Image.getcolors(colimg)}:
                #colours.add(colour)
                if colour in colours.keys():
                    colours[colour] += 1
                else:
                    colours[colour] = 1
            grayimg = Image.open(patientName + "\\" + videoName + "\\" + imageName).convert("L")
            for colour in {colour[1] for colour in Image.Image.getcolors(grayimg)}:
                # colours.add(colour)
                if colour in graycolours.keys():
                    graycolours[colour] += 1
                else:
                    graycolours[colour] = 1

                '''if colour in [(0, 0, 255), (0, 1, 255), (0, 150, 255),
                              (150, 0, 75), (150, 0, 255), (100, 50, 0),
                              (100, 51, 1), (101, 51, 0)
                              ] and colours[colour] % 50 == 0:
                    print(f"image:{imageName}, colour:{colour}")'''

print(colours)
print(graycolours)
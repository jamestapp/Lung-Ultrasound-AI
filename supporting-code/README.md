# Readme

This folder contains code used in the masters dissertation for applying deep learning techniques for lung ultrasound images. 

This project was undertaken by James Tapp while studying at Cardiff University, in co-operation with Intelligent Ultrasound. 

The code in dl-training-main was intended for use on the Intelligent Ultrasound GPU cluster, although may be freely modified to run on any other infrastructure. Within dl-training-main\source\UNet_Version_master is the code from the library for UNet 3+, with some additional loss functions. 

The code in onnx-inference is used for saving neural network models as .onnx files, and testing their runtime with the CPU execution provider. 

The code in data-analysis was used for gathering information about the lung ultrasound dataset in order to better understand how to best extract information and train a neural network model. 

The code in UNet_3Plus_DeepSup_CGM_MultiClass contains several modified versions of the UNet-3+ model, generalised to a multilabel classification problem. 
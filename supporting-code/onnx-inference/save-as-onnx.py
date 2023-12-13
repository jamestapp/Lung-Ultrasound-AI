import torch 

from UNet_Version_master.modelsReducedFilters.UNet_3Plus import UNet_3Plus as UNET

model = UNET(in_channels=1, n_classes=7)
fake_arg = torch.rand(1, 1, 256, 256)

torch.onnx.export(model, args=fake_arg, f="runtimetest-greyscale-256-256-unet3plus-reducedfilters.onnx")
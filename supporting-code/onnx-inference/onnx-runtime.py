import time
import onnxruntime
import numpy as np
import cv2
import albumentations as A
import torch

NUM_RUNS = 100
fake_img = np.random.rand(1, 1, 256, 256)
fake_img = fake_img.astype('float32')

options = onnxruntime.SessionOptions()

sess = onnxruntime.InferenceSession(f"runtimetest-greyscale-256-256-unet3plus-deepsup-reducedfilters.onnx", options, providers=['CPUExecutionProvider'])

times = np.empty((NUM_RUNS,))
for i in range(NUM_RUNS):
    start_time = time.time()
    x = [fake_img]
    feed = dict([(input.name, x[n]) for n, input in enumerate(sess.get_inputs())])
    pred_onnx = sess.run(None, feed)
    times[i] = time.time() - start_time

print(f"Processed successfully at an average of {1000 * times[10:].mean()} ms per image, stddev={1000 * times[10:].std()}")
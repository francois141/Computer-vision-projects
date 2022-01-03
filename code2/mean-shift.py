import time
import os
import random
import math
import torch
import numpy as np
import time


# run `pip install scikit-image` to install skimage, if you haven't done so
from skimage import io, color
from skimage.transform import rescale


def distance(x, X):
    distances = torch.zeros_like(torch.empty(X.shape[0]))
    for i in range(X.size(dim=0)):
        distances[i] = torch.sqrt(torch.sum(((x - X[i])**2)))

    return distances

def distance_batch(x, X):
    return torch.sqrt(torch.sum(((x.expand(X.shape[0],3) - X)**2),axis=1))

def gaussian(dist, bandwidth):
    tensor = torch.exp(-torch.pow(dist,2)/(2*bandwidth)) / (math.sqrt(2*math.pi)*bandwidth)
    return tensor

def update_point(weight, X):
    vector_unormalized = X[0]*weight[0]

    for i in range(1,weight.size(dim=0)):
        vector_unormalized += X[i]*weight[i]

    return vector_unormalized / torch.sum(weight)

def update_point_batch(weight, X):
    return torch.sum(torch.mul(weight.reshape(weight.shape[0],1).expand(weight.shape[0],3),X),axis=0) / torch.sum(weight)

def meanshift_step(X, bandwidth=2.5):
    X_ = X.clone()
    for i, x in enumerate(X):
        dist = distance(x, X)
        weight = gaussian(dist, bandwidth)
        X_[i] = update_point(weight, X)
    return X_

def meanshift_step_batch(X, bandwidth=2.5):
    X_ = X.clone()
    for i, x in enumerate(X):
        dist = distance_batch(x, X)
        weight = gaussian(dist, bandwidth)
        X_[i] = update_point_batch(weight, X)
    return X_

def meanshift(X):
    X = X.clone()
    for i in range(40):
        X = X.clone()
        start = time.time()
        print("Iteration ",i)
        X = meanshift_step(X)   # slow implementation
        #X = meanshift_step_batch(X)   # fast implementation
        end = time.time()
        print("Elasped time at iteration using batch {} : {}".format(i,end-start))  
    return X

scale = 0.25  # downscale the image to run faster

# Load image and convert it to CIELAB space
image = rescale(io.imread('cow.jpg'), scale, multichannel=True)
image_lab = color.rgb2lab(image)
shape = image_lab.shape # record image shape
image_lab = image_lab.reshape([-1, 3])  # flatten the image

# Run your mean-shift algorithm
t = time.time()
X = meanshift(torch.from_numpy(image_lab)).detach().cpu().numpy()
#X = meanshift(torch.from_numpy(image_lab).cuda()).detach().cpu().numpy()  # you can use GPU if you have one
t = time.time() - t
print ('Elapsed time for mean-shift: {}'.format(t))

# Load label colors and draw labels as an image
colors = np.load('colors.npz')['colors']
colors[colors > 1.0] = 1
colors[colors < 0.0] = 0




centroids, labels = np.unique((X / 10).round(), return_inverse=True, axis=0)
print(colors.shape)
print(labels)
result_image = colors[labels].reshape(shape)
result_image = rescale(result_image, 1 / scale, order=0, multichannel=True)     # resize result image to original resolution
result_image = (result_image * 255).astype(np.uint8)
io.imsave('result.png', result_image)

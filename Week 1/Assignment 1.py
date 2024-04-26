import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from pylab import ginput

### PROBLEM 1.1

A = np.random.rand(20, 20) * 256

plt.figure()
plt.imshow(A, vmin=0, vmax=256, cmap="gray")
plt.xticks(np.arange(20), np.arange(20))
plt.yticks(np.arange(20), np.arange(20))
plt.xlabel("X")
plt.ylabel("Y")
plt.colorbar()

coord = np.round(ginput(1))
print(f"Point chosen: {coord})")  # r,c

A[int(coord[0][1]), int(coord[0][0])] = 0  # x,y

plt.figure()
plt.imshow(A, vmin=0, vmax=256, cmap="gray")
plt.xticks(np.arange(20), np.arange(20))
plt.yticks(np.arange(20), np.arange(20))
plt.xlabel("X")
plt.ylabel("Y")
plt.colorbar()
plt.show()


### PROBLEM 1.2

B = imread("BBC_grey_testcard.png")
plt.imshow(B, vmin=0, vmax=256, cmap="gray")
plt.show()

bit_slices = []

for i in range(8):
    bit_i = np.bitwise_and(B, i+1)
    bit_slices.append(bit_i)

plt.figure(figsize=(10,5))

for i in range(8):
    plt.subplot(2,4,i+1)
    plt.imshow(bit_slices[i], cmap="gray")
    plt.title(f"bit no. {i+1}")

plt.tight_layout()
plt.show()

### PROBLEM 1.3

from skimage.color import rgb2hsv

C = imread("carpark.png")
C_hsv = rgb2hsv(C)

hue_C = C_hsv[:, :, 0]
saturation_C = C_hsv[:, :, 1]
value_C = C_hsv[:, :, 2]

fig, (ax0, ax1, ax2, ax3) = plt.subplots(ncols=4, figsize=(8, 2))

ax0.imshow(C)
ax0.set_title("RGB image")
ax0.axis('off')
ax1.imshow(hue_C)
ax1.set_title("Hue channel")
ax1.axis('off')
ax2.imshow(saturation_C)
ax2.set_title("Saturation channel")
ax2.axis("off")
ax3.imshow(value_C)
ax3.set_title("Value channel")
ax3.axis('off')

fig.tight_layout()
plt.show()


### PROBLEM 1.4

from skimage.util import img_as_float

def blend(A, B, w_A, w_B):
    A = img_as_float(A)
    B = img_as_float(B)

    C = A * w_A + B * w_B
    return C

cars1 = imread("toycars1.png")
cars2 = imread("toycars2.png")

cars_blended = blend(cars1, cars2, 0.3, 0.7)

plt.imshow(cars_blended)
plt.axis("off")
plt.show()


### PROBLEM 1.5

def RGB_to_gray(I):
    I_gray = 0.299 * I[:, :, 0] + 0.587 * I[:, :, 1] + 0.114 * I[:, :, 2]
    return I_gray

D = imread("carpark.png")

D_gray = RGB_to_gray(D)

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(D)
plt.axis("off")
plt.title("RGB image")

plt.subplot(1, 2, 2)
plt.imshow(D_gray, cmap="gray")
plt.axis("off")
plt.title("Gray scale image")

plt.show()


### PROBLEM 1.6

from skimage.transform import resize

E = imread("carpark.png")
print(f"The shape of the original image is {np.shape(E)}")

E_resized1 = resize(E, (200, 300))
E_resized2 = resize(E, (480/4, 640/4))
E_resized3 = resize(E, (480, 200))

plt.figure(figsize=(10,10))

plt.subplot(2, 2, 1)
plt.imshow(E)
plt.title("Original image")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(E_resized1)
plt.title("Resized image 1")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(E_resized2)
plt.title("Resized image 2")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(E_resized3)
plt.title("Resized image 3")
plt.axis("off")

plt.tight_layout()
plt.show()

import timeit

plt.figure(figsize=(12,20))

plt.subplot(4, 2, 1)
plt.imshow(E_resized2)
plt.title("Original image")
plt.axis("off")

E_enlarged0 = resize(E_resized2, (480*2, 640*2), order=0)

plt.subplot(4, 2, 2)
plt.imshow(E_enlarged0)
plt.title("Enlarged with method 0")
plt.axis("off")

E_enlarged1 = resize(E_resized2, (480*2, 640*2), order=1)

plt.subplot(4, 2, 3)
plt.imshow(E_enlarged1)
plt.title("Enlarged with method 1")
plt.axis("off")

E_enlarged2 = resize(E_resized2, (480*2, 640*2), order=2)

plt.subplot(4, 2, 4)
plt.imshow(E_enlarged2)
plt.title("Enlarged with method 2")
plt.axis("off")

E_enlarged3 = resize(E_resized2, (480*2, 640*2), order=3)

plt.subplot(4, 2, 5)
plt.imshow(E_enlarged3)
plt.title("Enlarged with method 3")
plt.axis("off")

E_enlarged4 = resize(E_resized2, (480*2, 640*2), order=4)

plt.subplot(4, 2, 6)
plt.imshow(E_enlarged4)
plt.title("Enlarged with method 4")
plt.axis("off")

E_enlarged5 = resize(E_resized2, (480*2, 640*2), order=5)

plt.subplot(4, 2, 7)
plt.imshow(E_enlarged5)
plt.title("Enlarged with method 5")
plt.axis("off")

plt.show()

# TODO: How to use timeit?


### PROBLEM 1.7

F = imread("railway.png")

plt.figure()
plt.imshow(F, vmin=0, vmax=256)
plt.xlabel("X")
plt.ylabel("Y")
plt.colorbar()

coord_foreground = np.round(ginput(2))
coord_background = np.round(ginput(2))

distance_foreground = np.abs(coord_foreground[0][0, 1] - coord_foreground[1][0, 1])

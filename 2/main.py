import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('./image1.jpg')
kernels = [
    np.array([[1, 1, 0],
              [1, 0, -1],
              [0, -1, -1]]),

    np.array([[1, 1, 1],
              [0, 0, 0],
              [-1, -1, -1]]),

    np.array([[0, 1, 1],
              [-1, 0, 1],
              [-1, -1, 0]]),

    np.array([[1, 0, -1],
              [1, 0, -1],
              [1, 0, -1]]),

    np.array([]),

    np.array([[-1, 0, 1],
              [-1, 0, 1],
              [-1, 0, 1]]),

    np.array([[0, -1, -1],
              [1, 0, -1],
              [1, 1, 0]]),

    np.array([[-1, -1, -1],
              [0, 0, 0],
              [1, 1, 1]]),

    np.array([[-1, -1, 0],
             [-1, 0, 1],
             [0, 1, 1]])
]
filters = []

fig, ax = plt.subplots(3, 3, figsize=(10, 6), sharex=True, sharey=True)
subplots = ax.flatten()

mult = float(input('Multiplier: '))

for i, kernel in enumerate(kernels):
    if i == 4:
        continue

    filter = cv2.filter2D(src=image, ddepth=-1, kernel=kernel * mult)
    filters.append(filter)
    subplots[i].imshow(filter)
    subplots[i].set_title(f'filter {i}')
    subplots[i].axis('off')

combined_filters = np.maximum.reduce(filters)

subplots[4].imshow(combined_filters)
subplots[4].set_title('combined filters')
subplots[4].axis('off')

plt.show()

import cv2
import matplotlib.pyplot as plt

img = cv2.imread('./image1.jpg', cv2.IMREAD_GRAYSCALE)

img_5 = img[:, :] & 0b11111000
img_4 = img[:, :] & 0b11110000
img_3 = img[:, :] & 0b11100000
img_2 = img[:, :] & 0b11000000
img_1 = img[:, :] & 0b10000000

fig, ax = plt.subplots(1, 5, figsize=(15, 5), sharex=True, sharey=True)
# plt.tight_layout()
subplots = ax.flatten()

subplots[0].imshow(img_5, 'gray')
subplots[0].set_title('5 bits')
subplots[0].axis('off')

subplots[1].imshow(img_4, 'gray')
subplots[1].set_title('4 bits')
subplots[1].axis('off')

subplots[2].imshow(img_3, 'gray')
subplots[2].set_title('3 bits')
subplots[2].axis('off')

subplots[3].imshow(img_2, 'gray')
subplots[3].set_title('2 bits')
subplots[3].axis('off')

subplots[4].imshow(img_1, 'gray')
subplots[4].set_title('1 bit')
subplots[4].axis('off')

plt.show()

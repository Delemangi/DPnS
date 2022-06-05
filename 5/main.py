import glob
import os

import cv2
import numpy as np


def load_images(dir: str) -> list[np.ndarray]:
    files = glob.glob(f'{dir}/*')
    images = []

    for file in files:
        images.append(cv2.imread(file, cv2.IMREAD_GRAYSCALE))

    return images


def sift(image: np.ndarray) -> tuple:
    sift = cv2.SIFT_create()
    return sift.detectAndCompute(image, None)


def get_matches(desc_1, desc_2) -> list:
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc_1, desc_2, k=2)
    good = []

    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])

    return good


def main() -> None:
    posters = load_images('posters')
    descriptors = [sift(image) for image in posters]

    img = input('Search image: ')
    image = cv2.imread(os.path.join('images', img), cv2.IMREAD_GRAYSCALE)
    kp, _ = sift(image)

    image_descriptors = [get_matches(desc[1], image) for desc in descriptors]
    best = image_descriptors.index(max(image_descriptors, key=len))

    poster_keypoints = cv2.drawKeypoints(posters[best], descriptors[best][0], None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    image_keypoints = cv2.drawKeypoints(image, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    image_1 = np.concatenate((image, posters[best]), axis=1)
    image_2 = np.concatenate((image_keypoints, poster_keypoints), axis=1)

    cv2.imshow('a', image_1)
    cv2.imshow('b', image_2)
    cv2.imshow('c', cv2.drawMatchesKnn(image, kp, posters[best], descriptors[best][0], image_descriptors[best], None, flags=2))

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

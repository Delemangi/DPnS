import os

import cv2
import numpy as np

def get_countours(img):
    image = cv2.imread(img)
    greyscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred_image = cv2.GaussianBlur(greyscale_image, (5, 5), 0)
    _, threshold_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    morphed_image = cv2.morphologyEx(threshold_image, cv2.MORPH_OPEN, kernel=np.ones((5, 5)), iterations=1)
    inverted_image = 255 - morphed_image

    contours, _ = cv2.findContours(inverted_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    final_image = cv2.drawContours(np.zeros(inverted_image.shape, np.uint8), contours, -1, 255, 1)

    cv2.imwrite(os.path.join('./results', os.path.basename(img)), final_image)

    return contours[0]


def main():
    similarities = {}
    input_folder = './images/'
    query_folder = './query/'

    query_image = get_countours(os.path.join(query_folder, input('Image: ')))
    images = [os.path.join(input_folder, i)
              for i in os.listdir(input_folder)
              if os.path.isfile(os.path.join(input_folder, i))]

    for image in images:
        contours = get_countours(image)
        similarities[os.path.basename(image)] = cv2.matchShapes(query_image, contours, 1, 0)

    for k, v in sorted(similarities.items(), key=lambda x: x[1]):
        print(f'{k}:\t{v}')


if __name__ == '__main__':
    main()

# се применуваат други алгоритми пред добивање на контури за повисок квалитет
# контурите се зачувувани во фолдерот results
# генерално споредба на лист со дршка (стебло) со лист без дава многу мала сличност и обратно (голем број враќа функцијата matchShapes)
# функцијата враќа број блиску до 0 за сличните листови, а враќа 0 за споредба меѓу едната иста слика
# резултатите се печатат во редослед на сличност, опаѓачки

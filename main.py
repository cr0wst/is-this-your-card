import os
import pathlib

import cv2
import imutils
import numpy as np

TRAINING_IMAGE_PATH = 'images/train/cards'
MIN_THRESHOLD = 100
MAX_THRESHOLD = 175


def main():
    cap = cv2.VideoCapture(0)
    training_set = load_training_set()
    cv2.namedWindow("Image")
    cv2.createTrackbar("Min Threshold", "Image", 100, 255, update_min_threshold)
    cv2.createTrackbar("Max Threshold", "Image", 175, 255, update_max_threshold)
    while True:
        global MIN_THRESHOLD
        global MAX_THRESHOLD
        MIN_THRESHOLD = cv2.getTrackbarPos("Min Threshold", "Image")
        MAX_THRESHOLD = cv2.getTrackbarPos("Max Threshold", "Image")
        ret, frame = cap.read()
        process(frame, training_set)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def process(frame, training_set):
    frame = imutils.resize(frame, 640)
    image = frame.copy()
    dilate = process_webcam_frame(image)
    contours = get_contours(dilate)

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        sides = len(approx)

        if 3 < sides <= 6:
            ((x, y), (w, h), a) = rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(image, [box], -1, (0, 255, 0), 3)
            # sometimes w and h are reversed for turned images
            ar = w / float(h) if w > h > 0 else h / float(w)

            if 1.30 <= ar <= 1.42:
                cv2.drawContours(image, [box], -1, (0, 255, 0), 3)
                cropped_image = crop_image(image, rect)
                if cropped_image.shape[0] > 0 and cropped_image.shape[1] > 0:
                    cropped_image = process_card_image(cropped_image)
                    cropped_image = resize_image(cropped_image)
                    prediction, percentage = predict(training_set, cropped_image)
                    cv2.putText(image, prediction, (int(x - w / 2), int(y)), cv2.FONT_HERSHEY_SIMPLEX, .75, 255, 4)
                    cv2.putText(image, str(f'{percentage * 100:.2f}' + '%'), (int(x - w / 2), int(y + 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, .50, 255, 2)
    cv2.imshow("Image", np.hstack((image, cv2.cvtColor(dilate, cv2.COLOR_RGB2BGR))))


def update_hough_threshold(value):
    global HOUGH_THRESH
    HOUGH_THRESH = value


def update_min_threshold(value):
    global MIN_THRESHOLD
    MIN_THRESHOLD = value


def update_max_threshold(value):
    global MAX_THRESHOLD
    MAX_THRESHOLD = value


def crop_image(original_image, rect):
    ((x, y), (w, h), a) = rect

    # Rotate the image so that the rectangle is in the same rotation as the frame.
    shape = (original_image.shape[1], original_image.shape[0])
    matrix = cv2.getRotationMatrix2D(center=(x, y), angle=a, scale=1)
    rotated_image = cv2.warpAffine(src=original_image, M=matrix, dsize=shape)

    # Crop the image from the rotation
    cx = int(x - w / 2)
    cy = int(y - h / 2)
    return rotated_image[cy:int(cy + h), cx:int(cx + w)]


def resize_image(image):
    approx = np.array(
        [[0, 0], [0, image.shape[0]], [image.shape[1], image.shape[0]],
         [image.shape[1], 0]], np.float32)
    h = np.array([[0, 0], [0, 449], [449, 449], [449, 0]], np.float32)
    transform = cv2.getPerspectiveTransform(np.float32(approx), h)
    return cv2.warpPerspective(image, transform, (450, 450))


def get_contours(dilate):
    contours = cv2.findContours(dilate.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    return sorted(contours, key=cv2.contourArea, reverse=True)


def predict(training_set, test_image):
    differences = []
    for training_image in training_set:
        diff = cv2.absdiff(test_image, training_image['image'])
        differences.append(np.sum(diff))

    match = training_set[np.argmin(differences)]

    index = np.argmin(differences)
    submatch = training_set[index]['image']
    diff = cv2.absdiff(test_image, submatch)

    # Calculate how close we are to matching by dividing the number of black pixels
    # by the total number of pixels. Black pixels indicate a match between the two images.
    total_pixels = diff.shape[0] * diff.shape[1]
    black_pixels = cv2.countNonZero(diff)
    percentage = (total_pixels - black_pixels) / total_pixels
    cv2.imshow("Matched", np.hstack((test_image, submatch, diff)))

    return match['label'], percentage


def load_training_set():
    training_set = []
    image_paths = load_all_image_paths()

    for path in image_paths:
        label = pathlib.Path(path).parent.name
        image = cv2.imread(path)
        image = process_card_image(image)
        contours = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]

        card = contours[0]
        peri = cv2.arcLength(card, True)
        approx = cv2.approxPolyDP(card, 0.02 * peri, True)
        h = np.array([[0, 0], [0, 449], [449, 449], [449, 0]], np.float32)
        transform = cv2.getPerspectiveTransform(np.float32(approx), h)
        warp = cv2.warpPerspective(image, transform, (450, 450))

        # Save Original
        training_set.append(dict(label=label, image=warp.copy()))

        # Rotate 3 times and save each to allow for non-symmetric symbols
        m = cv2.getRotationMatrix2D((warp.shape[1] / 2, warp.shape[0] / 2), 90, 1)

        warp = cv2.warpAffine(warp.copy(), m, (warp.shape[1], warp.shape[0]))
        training_set.append(dict(label=label, image=warp.copy()))

        warp = cv2.warpAffine(warp.copy(), m, (warp.shape[1], warp.shape[0]))
        training_set.append(dict(label=label, image=warp.copy()))

        warp = cv2.warpAffine(warp.copy(), m, (warp.shape[1], warp.shape[0]))
        training_set.append(dict(label=label, image=warp.copy()))
    return training_set


def load_labels():
    labels = []
    for directory in os.listdir(TRAINING_IMAGE_PATH):
        if directory[0] != '.':
            labels.append(directory)

    return labels


def load_all_image_paths():
    all_image_paths = []
    for path, subdirs, files in os.walk(TRAINING_IMAGE_PATH):
        for name in files:
            if name[-3:] == 'jpg':
                all_image_paths.append(os.path.abspath(os.path.join(path, name)))

    return all_image_paths


def label_images(image_paths, labels):
    label_to_index = dict((name, index) for index, name in enumerate(labels))
    return [label_to_index[pathlib.Path(path).parent.name] for path in image_paths]


def process_webcam_frame(frame):
    kernel = np.ones((5, 5), np.uint8)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, MIN_THRESHOLD, MAX_THRESHOLD)
    dilate = cv2.dilate(edges, kernel, iterations=1)

    return dilate


def process_card_image(image):
    gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    return thresh


main()

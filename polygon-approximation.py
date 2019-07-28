import cv2
import imutils
import numpy as np

def main():
    cap = cv2.VideoCapture(1)
    while True:
        ret, frame = cap.read()
        frame = imutils.resize(frame, 640)
        process(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def process(frame):
    image = frame.copy()
    dilate = normalize(frame)
    contours = get_contours(dilate)

    for c in contours[:2]:
        ((x, y), (w, h), a) = rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # sometimes w and h are reversed for turned images
        ar = w / float(h) if w > h > 0 else h / float(w)
        if 1.40 <= ar <= 1.45:
            cv2.drawContours(image, [box], -1, (0, 255, 0), 3)
                
    cv2.imshow("Image", image)

# Normalize the image by doing the following:
#   - Convert to Greyscale
#   - Apply Bilateral Filtering
#   - Find the Edges
#   - Dilate the Edges
def normalize(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 50, 150, True)
    kernel = np.ones((5, 5), np.uint8)
    dilate = cv2.dilate(edges, kernel, iterations=1)

    return dilate

# Connect all of the lines in the dilated image.
def get_contours(dilate):   
    contours = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    return sorted(contours, key=cv2.contourArea, reverse=True)

main()
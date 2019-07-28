import cv2
import imutils
import numpy as np

def main():
    cap = cv2.VideoCapture(1)
    while True:
        ret, frame = cap.read()
        frame = normalize(frame)
        # cv2.imshow("Image", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Normalize the image by doing the following:
# 	- Convert to Greyscale
#	- Apply Bilateral Filtering
#	- Find the Edges
#	- Dilate the Edges
def normalize(frame):
    frame = imutils.resize(frame, 640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 10, 15, 15)
    cv2.imshow("gray", gray)
    cv2.imshow("blur", blur)

    edges = cv2.Canny(blur, 50, 150, True)
    cv2.imshow("edges", edges)
    kernel = np.ones((5, 5), np.uint8)
    dilate = cv2.dilate(edges, kernel, iterations=1)

    cv2.imshow("dilated", dilate)

    return dilate

main()
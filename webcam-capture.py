import cv2
import imutils

def main():
    cap = cv2.VideoCapture(1)
    while True:
        ret, frame = cap.read()
        frame = imutils.resize(frame, 640)
        cv2.imshow("Image", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

main()
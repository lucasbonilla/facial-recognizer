import cv2

import utils as ut

def main():
    cam = cv2.VideoCapture(ut.CAMURL)
    while True:
        ret, face = cam.read()
        cv2.imshow("Entrada", cv2.flip(face, 1))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

if __name__ == '__main__':
	main()

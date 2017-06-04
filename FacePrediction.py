import cv2

import Capture
import FaceDB
import Utils as ut

model = cv2.face.createLBPHFaceRecognizer()
model.load(ut.MODELFILE)

# Retorna o nome e a distância da predição.
# Menor valor melhor a confiança
def predict(face):
    prediction, conf = model.predict(face)
    result = FaceDB.Label.get(FaceDB.Label.id == prediction).name
    return result, conf


def main1():
    cam = cv2.VideoCapture(0)
    while True:
        ret, face = cam.read()
        rects, captured = Capture.detect_faces(face)
        if captured:
            face = Capture.crop_face(rects, face)
            face = Capture.resize_face(face)
            print(predict(face))


if __name__ == '__main__':
    main1()

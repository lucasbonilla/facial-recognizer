import cv2

import Capture
import FaceDB
import Utils as ut

model = cv2.face.createLBPHFaceRecognizer()
model.load(ut.MODELFILE)

exemplo1 = '/home/lucas/PycharmProjects/facerecognitionLBPH/database/images/Lucas Boniila/Lucas Boniila_1.jpg'
exemplo2 = '/home/lucas/PycharmProjects/facerecognitionLBPH/database/images/Usuário Teste 70/Usuário Teste 70_459.jpg'
exemplo3 = '/home/lucas/PycharmProjects/facerecognitionLBPH/database/images/Usuário Teste 138/Usuário Teste 138_90.jpg'
exemplo4 = '/home/lucas/PycharmProjects/facerecognitionLBPH/database/images/Usuário Teste 202/Usuário Teste 202_143.jpg'
exemplo5 = '/home/lucas/PycharmProjects/facerecognitionLBPH/database/images/Usuário Teste 246/Usuário Teste 246_33.jpg'


# Retorna o nome e a distância da predição.
# Menor valor melhor a confiança
def predict(face):
    prediction, conf = model.predict(face)
    result = FaceDB.Label.get(FaceDB.Label.id == prediction).name
    return result, conf


def main2():
    print(predict(cv2.cvtColor(cv2.imread(exemplo1),
                       cv2.COLOR_BGR2GRAY)))


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

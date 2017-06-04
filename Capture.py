import os

import cv2

import Utils as ut
import time as t
import shutil as s
import re

frontal_face = cv2.CascadeClassifier(ut.CASCADEPATHFRONTALFACE)


# Recorta a face da imagem
def crop_face(rect, frame):
    return cv2.cvtColor(frame[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]],
                        cv2.COLOR_BGR2GRAY)


# Redimensiona a imagem em 100x100
def resize_face(frame):
    return cv2.resize(frame, (ut.IM_SIZE))


# Verifica qual a imagem mais próxima da câmera através da área que ela representa em pixels
def analyse(rects):
    area = 0
    index = -1
    if len(rects) == 0:
        return []
    for i in range(len(rects)):  # 0x, 1y, 2w, 3h
        area_image = rects[i][2] * rects[i][3]
        if area_image > area:
            area = area_image
            index = i
    return rects[index]


# Reconhece as faces que contém na imagem e retorna um booleano que confirma a detecção de uma face
def detect_faces(img):
    rects = frontal_face.detectMultiScale(img, scaleFactor=1.05, minNeighbors=6,
                                          minSize=(50, 50),
                                          maxSize=(500, 500),
                                          flags=cv2.CASCADE_SCALE_IMAGE)
    rects = analyse(rects)
    return rects, False if len(rects) == 0 else True


def input_name():
    dirs = os.listdir(ut.IMAGESPATH)
    directory = [re.split(r'(\d+)', sa) for sa in dirs]
    print(directory)
    index = 0
    for i in range(len(directory)):
        if directory[i][0] == 'convidado':
            if int(directory[i][1]) > int(index):
                index = directory[i][1]
    return int(index) + 1


def capture():
    i = 0
    name = input('Nome: ')

    new_path = ut.IMAGESPATH + '/' + name
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    cam = cv2.VideoCapture(0)

    time_exceeded = False
    t1 = t.time()

    while i < 10:
        ret, frame = cam.read()
        # print('diferenca:' + str(t.time() - t1))
        if t.time() - t1 > 5.0:
            print('era isso')
            print('Removendo path: ' + new_path)
            s.rmtree(new_path)
            time_exceeded = True
            break
        rect, is_captured = detect_faces(frame)
        if len(rect) != 0:
            frame = crop_face(rect, frame)
            frame = resize_face(frame)
        if is_captured:
            # print("deu")
            cv2.imwrite(new_path + '/' + name + '_' + str(i) + '.jpg', frame)
            i += 1
            t1 = t.time()
    return time_exceeded


if __name__ == '__main__':
    print("Running Capture.py.")
    capture()

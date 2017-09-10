import time as t

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


def main():
    confiability = []
    last = ""
    recognize = True
    while True:
        cam = cv2.VideoCapture(ut.CAMURL)
        while True:
            ret, full_img = cam.read()

            rects, captured = Capture.detect_faces(
                full_img)  # Recebe as coordenadas da face e um booleano
            if captured:  # Se capturou uma face
                face = Capture.crop_face(rects, full_img)  # Recorta
                face = Capture.resize_face(face)  # Redimensiona
                # cv2.imshow("ENTRADA", cv2.flip(face, 1))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                res, conf = predict(face)  # Predição
                if conf <= 90:  # Confiança menor igual que 90 é um positivo
                    print("Nome: %s, Confiança: %s" % (res, conf))
                    if last != res:
                        last = res
                    # print('Flush')
                    confiability.clear()

                    # for i in range(10):
                    path_recognized = ut.IMAGESPATHFINAL + "/" + res + "/" + res + "_0" + \
                                      ".jpg"
                    # print(path_recognized)
                    # recognized = cv2.imread(path_recognized)
                    cv2.putText(full_img, res, (rects[0], rects[1] - 10),
                                cv2.FONT_HERSHEY_PLAIN, 2, (0, 153, 255), 1)
                    cv2.rectangle(full_img, (rects[0], rects[1]), (rects[0]+rects[2],
                                  rects[1]+rects[3]), (0, 153, 255), 2)
                    cv2.imshow(res, full_img)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                    continue
                else:  # Confiança maior que 60 pode ser um positivo mas não é confiável
                    # print("Nome: %s, Confiança: %s" % (res, conf))
                    confiability.append((res, conf, t.time()))
                    index = len(confiability) - 1

                    # Se a última detecção de face for superior a 2 segundos ele ignora pois pode ser uma nova pessoa
                    if index != 0 and confiability[index][2] - confiability[index - 1][
                        2] > 2.0:
                        # print('Flush')
                        confiability.clear()
                    # Se reconheceu 10 imagens seguidas com confiabilidade menor que 60 por dez vezes seguidas
                    # entende-se que a base não conhece a face
                    if len(confiability) > 20:
                        # print("I don't know you...")
                        recognize = False  # Seta um False
                        confiability.clear()
                        break

        cv2.destroyAllWindows()

        # Se não reconheceu
        if not recognize:
            cam.release()
            time_exceeded, name, cpf = Capture.capture()
            if time_exceeded:
                continue
            FaceDB.atualize_db(name, cpf)
            FaceDB.update_model()
            recognize = True


if __name__ == '__main__':
    main()

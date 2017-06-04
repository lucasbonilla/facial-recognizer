import os, shutil

path = '/home/lucas/PycharmProjects/facerecognitionLBPH/database/images'
# pathd = '/home/lucas/PycharmProjects/facerecognitionLBPH/database/ytfaces/s'

caminhos = [os.path.join(path, nome) for nome in os.listdir(path)]


for c1 in caminhos:
    caminhos2 = [os.path.join(c1, nome) for nome in os.listdir(c1)]
    print('Apagando imagens de ' + c1)
    apagar = (len(caminhos2) - 10)
    for c2 in caminhos2:
        if apagar <= 0:
            break
        os.remove(c2)
        apagar -= 1

        # def main():
        #     cam = cv2.VideoCapture(0)
        #     while True:
        #         ret, face = cam.read()
        #         rects, captured = Capture.detect_faces(face)
        #         if captured:
        #             face = Capture.crop_face(rects, face)
        #             face = Capture.resize_face(face)
        #             print(predict(face))


# j = 61
# for c in caminhos:
#     os.makedirs(pathd + str(j))
#     print(c)
#     print(pathd + str(j))
#     caminhos2 = [os.path.join(c, nome) for nome in os.listdir(c)]
#     i = 1
#     for c2 in caminhos2:
#         caminhos3 = [os.path.join(c2, nome) for nome in os.listdir(c2)]
#         for cn in caminhos3:
#             # print('dest: '+pathd+str(j)+'/'+str(i)+'.jpg')
#             os.rename(cn, pathd+str(j)+'/'+str(i)+'.jpg')
#             i += 1
#     j += 1
        #     shutil.move(cn, c)
        # os.removedirs(c2)

# for j in range(1595):
#     os.system('cd ' + path + str(j+1) + '; ls ..; x=1; for i in *; do mv $i $x.jpg ; '
#                                         'x=$((x+1)); done')
#     break

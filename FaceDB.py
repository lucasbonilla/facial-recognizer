import os
import shutil

import cv2
import numpy as np
import peewee
import time as t

import Utils as ut

DB = peewee.SqliteDatabase(ut.DBPATH)


def save(path, img):
    cv2.imwrite(path, img)


def update_model():
    images, labels = load_images_from_db()
    model = cv2.face.createLBPHFaceRecognizer(threshold=200.0)
    model.update(images, labels)


# Realiza o treino da base
def train():
    images, labels = load_images_from_db()
    model = cv2.face.createLBPHFaceRecognizer(threshold=200.0)
    print("Treinando a base")
    model.train(images, labels)

    model.save(ut.MODELFILE)
    print("It's done!")


# Carrega o path para as tabelas
def load_images_to_db():
    for dir_name, dir_names, file_names in os.walk(ut.IMAGESPATH, topdown=False):
        for sub_dir_name in dir_names:
            subject_path = os.path.join(dir_name, sub_dir_name)
            label, p = Label.get_or_create(name=sub_dir_name)
            label.save()
            for filename in os.listdir(subject_path):
                path = os.path.abspath(os.path.join(subject_path, filename))
                # print('saving path %s' % path)
                image, p = Image.get_or_create(path=path, label=label)
                image.save()


# Carrega o path do banco
def load_images_from_db():
    images, labels = [], []
    for label in Label.select():
        for image in label.image_set:
            # print(label.id)
            # print(image.path)
            try:
                cv_image = cv2.imread(image.path, cv2.IMREAD_GRAYSCALE)
                images.append(np.asarray(cv_image, dtype=np.uint8))
                labels.append(label.id)
            except IOError as err:
                errno, strerror = err.args
                print("IOError({0}): {1}".format(errno, strerror))
    return images, np.asarray(labels)


class BaseModel(peewee.Model):
    class Meta:
        database = DB


class Label(BaseModel):
    name = peewee.CharField()

    def persist(self):
        path = os.path.join(ut.IMAGESPATH, self.name)
        if os.path.exists(path) and len(os.listdir(path)) >= 10:
            shutil.rmtree(path)
        if not os.path.exists(path):
            print('Created directory: %s' % self.name)
            os.makedirs(path)
        Label.get_or_create(name=self.name)


class Image(BaseModel):
    path = peewee.CharField()
    label = peewee.ForeignKeyField(Label)

    def persist(self, cv_image):
        path = os.path.join(ut.IMAGESPATH, self.label.name)
        nr_of_images = len(os.listdir(path))
        if nr_of_images >= 10:
            return 'Done'
        path += '/%s.jpg' % nr_of_images
        path = os.path.abspath(path)
        print('Saving %s' % path)
        cv2.imwrite(path, cv_image)
        self.path = path
        self.save()


def atualize_db():
    Image().delete()
    Label().delete()
    print('Iniciando Load')
    ini = t.time()
    load_images_to_db()
    print(t.time() - ini)
    print('Terminado Load')


if __name__ == '__main__':
    print('Iniciando')
    if os.path.isfile(ut.MODELFILE) is False:
        with open(ut.MODELFILE, "a") as arq:
            arq.close()
    print('Atualizando base de dados')
    atualize_db()
    ini = t.time()
    print('Treinamento')
    train()
    print(t.time() - ini)

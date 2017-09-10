import os
import shutil

import cv2
import numpy as np
import peewee
import time as t
import geradorcpf

import Utils as ut

DB = peewee.SqliteDatabase(ut.DBPATH)


def save(path, img):
    cv2.imwrite(path, img)


def update_model():
    images, labels = load_images_from_db()
    model = cv2.face.createLBPHFaceRecognizer(threshold=200.0)
    model.update(images, labels)
    model.save(ut.MODELFILE)


# Realiza o treino da base
def train():
    images, labels = load_images_from_db()
    model = cv2.face.createLBPHFaceRecognizer(threshold=200.0)
    print("Treinando a base")
    model.update(images, labels)

    model.save(ut.MODELFILE)
    print("It's done!")


# Carrega o path para as tabelas
def load_images_to_db(name_entrada, cpf_entrada):
    for dir_name, dir_names, file_names in os.walk(ut.IMAGESPATHPARTIAL, topdown=False):
        for sub_dir_name in dir_names:
            subject_path = os.path.join(dir_name, sub_dir_name)
            if len(name_entrada) is 0:
                name_entrada = sub_dir_name
            if len(cpf_entrada) is 0:
                cpf = geradorcpf.gerar()
            else:
                cpf = cpf_entrada
            label, p = Label.get_or_create(name=name_entrada, cpf=cpf)
            label.save()
            for filename in os.listdir(subject_path):
                finalpath = ut.IMAGESPATHFINAL+'/'+sub_dir_name
                path = os.path.abspath(os.path.join(finalpath, filename))
                image, p = Image.get_or_create(path=path, label=label)
                if not os.path.exists(finalpath):
                    os.makedirs(finalpath)
                shutil.move(subject_path+'/'+filename, finalpath+'/'+filename)
                image.save()
            shutil.rmtree(subject_path)


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
    cpf = peewee.IntegerField()

    def persist(self):
        path = os.path.join(ut.IMAGESPATHPARTIAL, self.name)
        if os.path.exists(path) and len(os.listdir(path)) >= 10:
            shutil.rmtree(path)
        if not os.path.exists(path):
            print('Created directory: %s' % self.name)
            os.makedirs(path)
        Label.get_or_create(name=self.name, cpf=self.cpf)


class Image(BaseModel):
    path = peewee.CharField()
    label = peewee.ForeignKeyField(Label)

    def persist(self, cv_image):
        path = os.path.join(ut.NEWIMAGESPATH, self.label.name)
        nr_of_images = len(os.listdir(path))
        if nr_of_images >= 10:
            return 'Done'
        path += '/%s.jpg' % nr_of_images
        path = os.path.abspath(path)
        print('Saving %s' % path)
        cv2.imwrite(path, cv_image)
        self.path = path
        self.save()


def atualize_db(name_entrada, cpf_entrada):
    # Image().delete()
    # Label().delete()
    print('Iniciando Load')
    ini = t.time()
    load_images_to_db(name_entrada, cpf_entrada)
    print('duracao: %f' % (t.time() - ini))
    print('Terminado Load')


if __name__ == '__main__':
    print('Iniciando')
    if os.path.isfile(ut.MODELFILE) is False:
        with open(ut.MODELFILE, "a") as arq:
            arq.close()
    print('Atualizando base de dados')
    atualize_db(name_entrada="", cpf_entrada="")
    ini = t.time()
    print('Treinamento')
    train()
    print(t.time() - ini)

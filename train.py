import os
import argparse
import numpy as np
from tqdm import tqdm
import pandas as pd
import joblib
from collections import OrderedDict

import tensorflow.keras
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, CSVLogger
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import cifar100

from utils import *
from wide_resnet import *
from LeNet import *

from cosine_annealing import *
from dataset import Cifar10ImageDataGenerator


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: resnet)')
    parser.add_argument('--depth', default=28, type=int)#resnet-28
    parser.add_argument('--width', default=10, type=int)
    parser.add_argument('--epochs', default=0, type=int)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--cutout', default=False, type=str2bool)
    parser.add_argument('--auto-augment', default=False, type=str2bool)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    if args.name is None:
        args.name = 'ResNet'
        if args.cutout:
            args.name += '_wCutout'
        if args.auto_augment:
            args.name += '_wAutoAugment'

    if not os.path.exists('models/%s' %args.name):
        os.makedirs('models/%s' %args.name)


    joblib.dump(args, 'models/%s/args.pkl' %args.name)


    #model=LeNet_5()
    #model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model=load_model('models/cifar100/lenet5_with_dropout.h5')

    (x_train, y_train), (x_test, y_test) = cifar100.load_data()

    datagen = Cifar10ImageDataGenerator(args)

    x_test = datagen.standardize(x_test)

    #y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 100)


    # callbacks = [
    #     ModelCheckpoint('models/lenet.h5', verbose=1, save_best_only=True),
    #     CSVLogger('log.csv'),
    # ]
    #
    #
    #
    # model.fit_generator(datagen.flow(x_train, y_train, batch_size=64),
    #                     steps_per_epoch=len(x_train)//args.batch_size,
    #                     validation_data=(x_test, y_test),
    #                     epochs=50, verbose=1,
    #                     callbacks=callbacks)

    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

if __name__ == '__main__':
    main()

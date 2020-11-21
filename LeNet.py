import keras
from keras.models import Sequential
from keras.layers import Input,Dense,Activation,Conv2D,MaxPooling2D,Flatten
from keras.datasets import cifar10
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, CSVLogger
from tensorflow.python.keras.models import load_model

(x_train,y_train),(x_test,y_test) = cifar10.load_data()
x_train = x_train.reshape(-1, 32, 32, 3)
x_train = x_train.astype("float32")
print(x_train.shape)
y_train = y_train.astype("float32")
x_test = x_test.reshape(-1,32,32,3)
x_test = x_test.astype("float32")
y_test = y_test.astype("float32")

print(y_train)
x_train /= 255
x_test /= 255

from keras.utils import np_utils
y_train_new = np_utils.to_categorical(num_classes=100,y=y_train)
print(y_train_new)
y_test_new = np_utils.to_categorical(num_classes=100,y=y_test)

def LeNet_5():
    model = Sequential()
    model.add(Conv2D(filters=6,kernel_size=(5,5),padding="valid",activation="tanh",input_shape=[32, 32, 3]))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(filters=16,kernel_size=(5,5),padding="valid",activation="tanh"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(120,activation="tanh"))
    model.add(Dense(84,activation="tanh"))
    model.add(Dense(10,activation="softmax"))
    return model

def train_model():
    model = LeNet_5()
    model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])
    #model=load_model('models/lenet.h5')
    callbacks = [
        ModelCheckpoint('lenet-100.h5', verbose=1, save_best_only=True),
        CSVLogger('log.csv'),
    ]
    model.fit(x_train,y_train_new,batch_size=64,epochs=100,verbose=1,validation_split=0.2,shuffle=True,callbacks=callbacks)
    return model

# model = train_model()
#
# loss,accuracy = model.evaluate(x_test,y_test_new)
# print(loss,accuracy)
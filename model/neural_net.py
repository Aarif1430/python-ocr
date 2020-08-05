import numpy as np
import keras
from keras import layers
from keras import models
from keras.utils import to_categorical

alphabets_mapper = {10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J',
                        20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T', 30: 'U',
                        31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z', 36: '5star', 37: 'scribble', 38: 'triangle',
                        39: ':)',40: ':(', 0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9'}


def load_sample():
    from sklearn.model_selection import train_test_split
    import pandas as pd
    datafile = '../data/data.csv'
    dataset = pd.read_csv(datafile)
    X = dataset.drop(['labels', 'Unnamed: 0'], axis=1).values
    Y = dataset['labels'].values
    my_dict2 = {y: x for x, y in alphabets_mapper.items()}
    data = []
    fn = lambda x: 0 if x < 200 else 255
    vfunc = np.vectorize(fn)
    for dat in X:
        d = np.asarray(255 - (np.float32(dat.reshape(20, 20) * 255)))
        data.append(vfunc(d))
    x = np.array(data)

    for i, item in enumerate(Y):
        if item in my_dict2.keys():
            Y[i] = my_dict2[item]

    x_train, x_test, y_train, y_test = train_test_split(x, Y)
    return (x_train, y_train), (x_test, y_test)



class NeuralNetwork(object):

    def __init__(self):
        self.model = models.Sequential()
        self.model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(20, 20, 1)))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(64, activation='relu'))
        self.model.add(layers.Dense(41, activation='softmax'))

    def train(self):
        (train_images, train_labels), (test_images, test_labels) = load_sample()
        train_images = train_images.reshape(train_images.shape[0], 20, 20, 1)
        train_images = train_images.astype('float32')
        test_images = test_images.reshape(test_images.shape[0], 20, 20, 1)
        test_images = test_images.astype('float32')
        train_labels = to_categorical(train_labels)
        test_labels = to_categorical(test_labels)
        self.model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        self.model.fit(train_images, train_labels, epochs=10, batch_size=64)
        test_loss, test_acc = self.model.evaluate(test_images, test_labels)
        print('Test accuracy:', test_acc)

    def save(self):
        self.model.save('conv_cnn_2.h5')

    def pretrained_model(self, saved_model):
        predtrained_model = keras.models.load_model(saved_model)
        return predtrained_model


# cnn = NeuralNetwork()
# cnn.train()
# cnn.save()
# model = cnn.pretrained_model('conv_cnn2.h5')
# # print('')
# v=x[1000].reshape(1, 20, 20, 1)
# np.argmax(model.predict(v))
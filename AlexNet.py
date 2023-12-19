import tensorflow as tf
import keras
from keras import models, layers

# input shape = (227, 227, 3)
def AlexNet(input_shape, num_classes):

    model = models.Sequential()

    # Layer 1
    model.add(layers.Conv2D(96, (11, 11), strides=(4, 4), input_shape=input_shape, padding='valid'))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))
    model.add(layers.BatchNormalization())

    # Layer 2
    model.add(layers.Conv2D(256, (5, 5), strides=(1, 1), padding='valid'))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))
    model.add(layers.BatchNormalization())

    # Layer 3
    model.add(layers.Conv2D(384, (3, 3), strides=(1, 1), padding='valid'))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())

    # Layer 4
    model.add(layers.Conv2D(384, (3, 3), strides=(1, 1), padding='valid'))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())

    # Layer 5
    model.add(layers.Conv2D(256, (3, 3), strides=(1, 1), padding='valid'))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))
    model.add(layers.BatchNormalization())

    # Flatten the output for the fully connected layers
    model.add(layers.Flatten())

    # Fully connected layers
    # First Dense Layer
    model.add(layers.Dense(4096, activation='relu'))
    #model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.BatchNormalization())

    # Second Dense Layer
    model.add(layers.Dense(4096, activation='relu'))
    #model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.BatchNormalization())

    # Third Dense Layer
    model.add(layers.Dense(1000, activation='relu'))
    #model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.BatchNormalization())

    model.add(layers.Dense(num_classes, activation='softmax'))
    
    return model

Model = AlexNet((227, 227, 3), 17)
print(Model.summary())
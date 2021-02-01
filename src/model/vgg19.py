import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.keras.engine import training

def VGG19(classes, classifier_activation='softmax', img_size=256, data_format='channels_last'):
    img_input = layers.Input(shape=(img_size, img_size, 3))
    x = layers.Conv2D(
        64, (3, 3), activation='relu', padding='same', name='block1_conv1', data_format=data_format)(img_input)
    x = layers.Conv2D(
        64, (3, 3), activation='relu', padding='same', name='block1_conv2', data_format=data_format)(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', data_format=data_format)(x)

    # Block 2
    x = layers.Conv2D(
        128, (3, 3), activation='relu', padding='same', name='block2_conv1', data_format=data_format)(x)
    x = layers.Conv2D(
        128, (3, 3), activation='relu', padding='same', name='block2_conv2', data_format=data_format)(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', data_format=data_format)(x)

    # Block 3
    x = layers.Conv2D(
        256, (3, 3), activation='relu', padding='same', name='block3_conv1', data_format=data_format)(x)
    x = layers.Conv2D(
        256, (3, 3), activation='relu', padding='same', name='block3_conv2', data_format=data_format)(x)
    x = layers.Conv2D(
        256, (3, 3), activation='relu', padding='same', name='block3_conv3', data_format=data_format)(x)
    x = layers.Conv2D(
        256, (3, 3), activation='relu', padding='same', name='block3_conv4', data_format=data_format)(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', data_format=data_format)(x)

    # Block 4
    x = layers.Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block4_conv1', data_format=data_format)(x)
    x = layers.Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block4_conv2', data_format=data_format)(x)
    x = layers.Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block4_conv3', data_format=data_format)(x)
    x = layers.Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block4_conv4', data_format=data_format)(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool', data_format=data_format)(x)

    # Block 5
    x = layers.Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block5_conv1', data_format=data_format)(x)
    x = layers.Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block5_conv2', data_format=data_format)(x)
    x = layers.Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block5_conv3', data_format=data_format)(x)
    x = layers.Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block5_conv4', data_format=data_format)(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    x = layers.GlobalAveragePooling2D(name='flatten', data_format=data_format)(x)
    x = layers.Dense(512, activation='relu', name='fc1')(x)
    x = layers.Dropout(0.5, name='dropout1')(x)
    x = layers.Dense(512, activation='relu', name='fc2')(x)
    x = layers.Dropout(0.5, name='dropout2')(x)
    x = layers.Dense(classes, activation=classifier_activation,
                     name='predictions')(x)
    model = training.Model(img_input, x, name='vgg19')
    return model
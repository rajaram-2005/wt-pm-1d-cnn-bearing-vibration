"""1D-CNN for gearbox bearing-fault classification from raw vibration windows.

This module defines the architecture only: it ships no data loader, training
loop or trained weights. Run ``python model.py`` to build the reference
configuration and print its summary. The wt-pm platform imports this file as-is
and calls ``build_1d_cnn``, so keep that signature stable.
"""
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout

def build_1d_cnn(input_shape, num_classes):
    """Return a compiled Keras 1D-CNN classifier.

    input_shape: (samples, channels) of one vibration window, e.g. (1024, 1).
    num_classes: number of fault classes. The output layer is a softmax and the
        loss is categorical cross-entropy, so labels must be one-hot encoded
        (e.g. with tf.keras.utils.to_categorical).
    """
    model = tf.keras.Sequential([
        tf.keras.Input(shape=input_shape),
        Conv1D(32, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Conv1D(64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Conv1D(128, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


if __name__ == "__main__":
    # Smoke test: build the reference configuration (one 1024-sample window,
    # 4 classes) and print the architecture. Nothing is loaded or trained here.
    build_1d_cnn((1024, 1), num_classes=4).summary()


import tensorflow as tf
from tensorflow.keras import layers, backend as K

def precision_m(y_true, y_pred):
    y_true = K.one_hot(K.cast(y_true, 'int32'), num_classes)
    y_true = K.cast(y_true, 'float32')
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    pred_pos = K.sum(K.round(K.clip(y_pred, 0, 1)))
    return tp / (pred_pos + K.epsilon())

def recall_m(y_true, y_pred):
    y_true = K.one_hot(K.cast(y_true, 'int32'), num_classes)
    y_true = K.cast(y_true, 'float32')
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    actual_pos = K.sum(K.round(K.clip(y_true, 0, 1)))
    return tp / (actual_pos + K.epsilon())

def f1_m(y_true, y_pred):
    p = precision_m(y_true, y_pred)
    r = recall_m(y_true, y_pred)
    return 2 * ((p * r) / (p + r + K.epsilon()))

def create_model(input_shape, num_classes):
    return tf.keras.Sequential([
        layers.Conv2D(8, (1, 1), activation='relu', kernel_initializer='he_normal', input_shape=input_shape),
        layers.Conv2D(16, (10, 1), strides=(10, 1), activation='relu', kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.Conv2D(16, (10, 1), strides=(10, 1), activation='relu', kernel_initializer='he_normal'),
        layers.MaxPooling2D((1, 4)),
        layers.Conv2D(32, (1, 4), activation='relu', kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((1, 2)),
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation='softmax')
    ])

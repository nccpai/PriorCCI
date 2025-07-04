import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers

def configure_device(use_gpu=True):
    """
    Configure device to use GPU or CPU.

    Parameters
    ----------
    use_gpu : bool
        If True, enables GPU (if available); if False, forces CPU usage.
    """
    if not use_gpu:
        print("Using CPU only (GPU disabled).")
        tf.config.set_visible_devices([], 'GPU')
    else:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"Using GPU: {len(gpus)} available")
            except RuntimeError as e:
                print("Error setting GPU configuration:", e)
        else:
            print("No GPU found. Running on CPU.")

def precision_m(num_classes):
    def precision_fn(y_true, y_pred):
        y_true_oh = K.one_hot(K.cast(y_true, 'int32'), num_classes)
        y_true_oh = K.cast(y_true_oh, 'float32')
        tp = K.sum(K.round(K.clip(y_true_oh * y_pred, 0, 1)))
        pred_pos = K.sum(K.round(K.clip(y_pred, 0, 1)))
        return tp / (pred_pos + K.epsilon())
    return precision_fn

def recall_m(num_classes):
    def recall_fn(y_true, y_pred):
        y_true_oh = K.one_hot(K.cast(y_true, 'int32'), num_classes)
        y_true_oh = K.cast(y_true_oh, 'float32')
        tp = K.sum(K.round(K.clip(y_true_oh * y_pred, 0, 1)))
        actual_pos = K.sum(K.round(K.clip(y_true_oh, 0, 1)))
        return tp / (actual_pos + K.epsilon())
    return recall_fn

def f1_m(num_classes):
    def f1_fn(y_true, y_pred):
        p = precision_m(num_classes)(y_true, y_pred)
        r = recall_m(num_classes)(y_true, y_pred)
        return 2 * ((p * r) / (p + r + K.epsilon()))
    return f1_fn

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

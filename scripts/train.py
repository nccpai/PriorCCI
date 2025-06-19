
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras import callbacks
import tensorflow as tf
from .utils import create_model, f1_m, precision_m, recall_m

def train_and_save_model(data, labels, num_classes, num_lrpair,
                         base_save_path='cnn_model/', n_repeat=10, batch_size=32, epochs=80):
    os.makedirs(base_save_path, exist_ok=True)

    for i in range(1, n_repeat + 1):
        print(f"▶ Training model {i}/{n_repeat}")
        rs = 41 + i
        X_tmp, test_data, Y_tmp, test_labels = train_test_split(data, labels, test_size=0.2, random_state=rs)
        train_data, val_data, train_labels, val_labels = train_test_split(X_tmp, Y_tmp, test_size=0.2, random_state=rs + 1)

        model = create_model((100, num_lrpair, 2), num_classes)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy', f1_m, precision_m, recall_m])

        lr_reduction = callbacks.ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, min_lr=1e-6)

        model.fit(train_data, train_labels, validation_data=(val_data, val_labels),
                  epochs=epochs, batch_size=batch_size, callbacks=[lr_reduction], verbose=1)

        model.save(os.path.join(base_save_path, f"data_cnn-model_v0{i}.h5"))
        print(f"✔ Model {i} saved.")

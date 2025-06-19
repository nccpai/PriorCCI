
import os
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from .utils import f1_m, precision_m, recall_m

def evaluate_saved_models(data, labels, num_classes, model_dir='cnn_model/', n_repeat=10):
    for i in range(1, n_repeat + 1):
        print(f"=== Model v{i} Evaluation ===")
        rs = 41 + i
        _, test_data, _, test_labels = train_test_split(data, labels, test_size=0.2, random_state=rs)

        model_path = os.path.join(model_dir, f"data_cnn-model_v0{i}.h5")
        model = load_model(model_path, custom_objects={
            'f1_m': f1_m,
            'precision_m': precision_m,
            'recall_m': recall_m
        })

        y_pred_probs = model.predict(test_data, batch_size=32, verbose=0)
        pred_labels = np.argmax(y_pred_probs, axis=1)
        true_labels = test_labels.astype(int)

        f1 = f1_score(true_labels, pred_labels, average='weighted')
        recall = recall_score(true_labels, pred_labels, average='weighted')
        precision = precision_score(true_labels, pred_labels, average='weighted')
        loss, accuracy, *_ = model.evaluate(test_data, true_labels, verbose=0)

        print(f"Loss: {loss:.4f} | Accuracy: {accuracy:.4f} | F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")

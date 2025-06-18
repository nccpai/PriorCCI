import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle

def visualize_final_model_results(model, test_data, test_labels, class_names=None):
    predictions = model.predict(test_data)
    predicted_classes = np.argmax(predictions, axis=1)

    cm = confusion_matrix(test_labels, predicted_classes)
    n_classes = len(np.unique(test_labels))

    if class_names is None:
        class_names = [f'Class {i}' for i in range(n_classes)]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion matrix')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()

    y_test_bin = label_binarize(test_labels, classes=np.arange(n_classes))
    y_score = predictions

    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(10, 8))
    colors = cycle(['blue', 'red', 'green', 'cyan', 'magenta', 'yellow',
                    'black', 'orange', 'purple', 'brown', 'gray'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'{class_names[i]} (AUC = {roc_auc[i]:0.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('Multi-class ROC curve')
    plt.legend(loc='lower right', fontsize=8)
    plt.show()
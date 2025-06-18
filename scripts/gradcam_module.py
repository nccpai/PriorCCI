import os
import re
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import layers, Model
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore

def convert_sequential_to_functional(model):
    inputs = layers.Input(shape=model.input_shape[1:])
    x = inputs
    for layer in model.layers:
        x = layer(x)
    return Model(inputs, x)

def run_gradcam_analysis(data, gene_list_csv, model_dir='cnn_model/',
                         save_dir='gcam_res/', class_files=None,
                         custom_objects=None, data_points=1000):
    os.makedirs(save_dir, exist_ok=True)

    model_names = sorted([f for f in os.listdir(model_dir) if f.endswith('.h5')])
    class_names = [re.search(r'combi-(.*)_c\d+\.npz', file).group(1) for file in class_files]

    gene_df = pd.read_csv(gene_list_csv)
    genes_A = gene_df['A'].tolist()
    genes_B = gene_df['B'].tolist()

    for model_name in model_names:
        version_suffix = model_name.split('_')[-1].replace('.h5', '')
        model_path = os.path.join(model_dir, model_name)
        print(f"🔍 Loading model: {model_path}")

        model = load_model(model_path, custom_objects=custom_objects)
        model = convert_sequential_to_functional(model)
        gradcam = GradcamPlusPlus(model, model_modifier=ReplaceToLinear(), clone=True)

        for class_index, class_name in enumerate(class_names):
            start = class_index * data_points
            end = start + data_points
            class_data = data[start:end]
            class_labels = np.full((data_points,), class_index)

            cam_tot = np.mean([
                gradcam(CategoricalScore(label), np.expand_dims(sample, axis=0), penultimate_layer=-1)
                for label, sample in zip(class_labels, class_data)
            ], axis=0)

            cam_mean = np.mean(cam_tot[0], axis=0)
            cam_norm = (cam_mean - cam_mean.min()) / (cam_mean.max() - cam_mean.min())

            output_txt = os.path.join(save_dir, f"gcamplus_result_{class_name}_{version_suffix}.txt")
            with open(output_txt, 'w') as f_out:
                f_out.write(f"TumorCell\t{class_name}\tNormalized_Weight\n")
                for a, b, w in zip(genes_A, genes_B, cam_norm):
                    f_out.write(f"{a}\t{b}\t{w}\n")

            print(f"✅ Completed: {output_txt}")
import tensorflow as tf
import numpy as np
import pandas as pd
import sys
import seaborn as sns
import os
import matplotlib.pyplot as plt

from datetime import datetime
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm

sys.path.append("../diploma_script")
from helpers.logger import Logger as logger


def build_model(input_layer, output_shape, n_hidden_layers=1, n_units=64, dropout_rate=0.1):
    logger.info(f"Building model with n_hidden_layer={n_hidden_layers}, "
                f"n_units={n_units} and dropout_rate={dropout_rate}")
    model = tf.keras.Sequential([
        input_layer,
    ])
    for _ in range(n_hidden_layers):
        model.add(tf.keras.layers.Dense(units=n_units, activation=tf.nn.relu))
        model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.Dense(units=output_shape, activation=tf.nn.softmax))
    logger.ok("Model built")
    return model


def compute_loss(y_true, y_pred):
    loss = tf.keras.losses.categorical_crossentropy(y_true=y_true, y_pred=y_pred)
    return loss


@tf.function
def train_step(model, optimizer, x, y):
    with tf.GradientTape() as tape:
        y_pred = model(x)
        loss = compute_loss(y_true=y, y_pred=y_pred)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss


def train_model(model, train_features, train_labels, num_training_iterations, batch_size, learning_rate):
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    losses = []

    logger.info(f"Training model with num_training_iterations={num_training_iterations}, batch_size={batch_size} "
                f"and learning_rate={learning_rate}...")
    for _ in tqdm(range(num_training_iterations)):
        sample = np.random.randint(0, train_features.shape[0], size=batch_size)
        x_batch, y_batch = train_features.iloc[sample].to_numpy(), train_labels[sample]
        loss = train_step(model, optimizer, x_batch, y_batch)
        losses.append(loss.numpy().mean())

    logger.ok("Training finished.")
    plt.figure(figsize=(10, 7))
    sns.lineplot(x=range(0, num_training_iterations), y=losses)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.show()

    return model


def adapt_normalizer(train_features):
    logger.info("Adapting normalization layer")
    normalization_layer = tf.keras.layers.Normalization(input_shape=(train_features.shape[1],), axis=-1)
    normalization_layer.adapt(np.array(train_features))
    logger.ok("Adapt finished.")

    return normalization_layer


def split_data(features, labels):
    encoder = OneHotEncoder()
    encoded_labels = encoder.fit_transform(labels.to_numpy().reshape(-1, 1)).toarray()
    np.save("classes.npy", encoder.categories_)
    logger.info("Splitting data...")
    train_features, test_features, train_labels, test_labels = train_test_split(features, encoded_labels, test_size=0.2, random_state=42)
    logger.info(f"Train Features: {train_features.shape}; Test Features: {test_features.shape}")
    logger.info(f"Train Labels: {train_labels.shape}; Test Labels: {test_labels.shape}")
    return train_features, test_features, train_labels, test_labels


def plot_loss_curves(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.plot(range(0, len(loss)), loss, label="Training loss")
    plt.plot(range(0, len(loss)), val_loss, label="Validation loss")
    plt.legend()


def plot_conf_matrix(conf_matr, labels):
    d = pd.DataFrame(np.array(conf_matr, dtype=np.float32), index=labels, columns=labels)
    plt.figure(figsize=(17, 14), dpi=60)
    sns.set(font_scale=1.4)
    sns.heatmap(d, annot=True, fmt="g")
    plt.xlabel("Output Class")
    plt.ylabel("Target Class")
    plt.tight_layout()
    plt.show()


def save_model(model):
    filepath = "models/model-{}".format(datetime.now().strftime("%d.%m.%Y-%H.%M.%S"))
    logger.info("Saving model to {}".format(os.getcwd() + "/" + filepath))
    tf.keras.models.save_model(model, filepath)
    logger.ok("Model saved.")

    return filepath


def run_training(features, labels):
    train_features, test_features, train_labels, test_labels = split_data(features, labels)
    normalization_layer = adapt_normalizer(train_features)
    encoder = OneHotEncoder()
    encoder.categories_ = np.load("classes.npy", allow_pickle=True)

    model = build_model(input_layer=normalization_layer, output_shape=len(np.unique(labels)), n_hidden_layers=10,
                        n_units=64, dropout_rate=0.5)

    model = train_model(model, train_features, train_labels, num_training_iterations=13000, batch_size=256,
                        learning_rate=2e-3)

    _ = save_model(model)

    conf_matrix = tf.math.confusion_matrix(labels=np.argmax(test_labels, axis=1),
                                           predictions=np.argmax(model.predict(test_features), axis=1))
    plot_conf_matrix(conf_matr=conf_matrix, labels=encoder.categories_.ravel().tolist())

    accuracy = np.mean(np.argmax(model.predict(test_features), axis=1) == np.argmax(test_labels, axis=1))
    logger.ok(f"Accuracy on the test set: {accuracy * 100}% ")

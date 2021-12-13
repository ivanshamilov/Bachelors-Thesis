import tensorflow as tf
import pandas as pd
import numpy as np
import socket 


from helpers.logger import Logger as logger

protocols = {
    0: "raw",
    6: "tcp",
    17: "udp"
}


def convert_dst_port(x, ports):
    result = "unassigned"
    try:
        result = socket.getservbyport(int(x['dst_port']), x['protocol'])
        if result not in ports:
            return "unassigned"
    except OSError:
        return "unassigned"

    return result


def align_df_with_training_set(df):
    logger.info("Aligning collected data with training data...")
    df['protocol'] = df['protocol'].map(protocols)

    training_headers = pd.read_csv("transformed_headers.csv")
    training_ports = [port.split("_")[2] for port in training_headers.filter(regex="dst_port").columns]
    df['dst_port'] = df.apply(lambda x: convert_dst_port(x, training_ports), axis=1)

    collected_ports = df.pop('dst_port')
    collected_protocols = df.pop('protocol')
    df, _ = df.align(training_headers, join="right", axis=1)

    df.drop('label', axis=1, inplace=True)
    df.fillna(0, inplace=True)

    for index, (port, protocol) in enumerate(zip(collected_ports.tolist(), collected_protocols.tolist())):
        df.iloc[index][f"dst_port_{port}"] = 1
        df.iloc[index][f"protocol_{protocol}"] = 1

    logger.ok("Alignment finished.")

    return df


def classify_flow(df, csv_path):
    # Path to pretrained model (sudo ./main.py -m train -d <path to dir with csv files>)
    model_path = "models/best_model"
    logger.info("Loading model...")
    model = tf.keras.models.load_model(model_path)
    logger.info(f"Classifying collected flows from {csv_path} with {model_path} model")
    predictions = model.predict(np.array(df))
    encoded_labels = np.load("classes.npy", allow_pickle=True)
    
    return [(encoded_labels.ravel()[np.argmax(prediction)], np.max(prediction)) for prediction in predictions]


def transformer_handler(csv_path):
    df = pd.read_csv(csv_path)
    collected_timestamps = df.pop('timestamp')
    df = align_df_with_training_set(df)
    predictions = classify_flow(df, csv_path)

    for t, (predicted_label, probability) in zip(collected_timestamps, predictions):
        if predicted_label == "Benign":
            logger.info(f"Is OK (probability: {probability * 100:.2f}%) at {t}")
        else:
            logger.warning(f"{predicted_label} (probability: {probability * 100:.2f}%) occurred at {t}")
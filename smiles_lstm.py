import json
from typing import Dict, List

import keras
import numpy as np
import tensorflow as tf


def input_preprocessing(fold: str) -> np.ndarray:
    with open(f"data/{fold}.txt", "r") as f:
        padded_smiles = [line.strip().split() for line in f.readlines()]

    label_encoded_smiles = np.expand_dims(
        np.array(
            [[TOKEN2LABEL[token] for token in molecule] for molecule in padded_smiles]
        ),
        axis=2,
    )
    X, y = (
        label_encoded_smiles[:, :-1, :],
        label_encoded_smiles[:, 1:, :],
    )
    return X, y


def build_model() -> keras.models.Sequential:
    clm = keras.models.Sequential(
        [
            keras.layers.Embedding(
                input_dim=VOCAB_SIZE,
                output_dim=EMBEDDING_DIM,
                input_length=TRAINING_LEN,
            ),
            keras.layers.LSTM(LSTM_SIZE, return_sequences=True),
            keras.layers.LSTM(LSTM_SIZE, return_sequences=True),
            keras.layers.TimeDistributed(
                keras.layers.Dense(VOCAB_SIZE, activation="softmax")
            ),
        ]
    )
    clm.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
    )
    return clm


def train_model(clm: keras.models.Sequential) -> Dict[str, List[float]]:
    history = clm.fit(
        x=X_train,
        y=y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=N_EPOCHS,
        verbose=1,
    ).history
    return history


def design_molecules(clm: keras.models.Sequential) -> List[str]:
    def temperature_sampling(preds, temperature):
        log_preds = np.log(preds.astype(np.float64) + 10e-10) / temperature
        return (
            tf.random.categorical(log_preds, num_samples=1).numpy().squeeze().tolist()
        )

    beg_label, pad_label = TOKEN2LABEL["<BEG>"], TOKEN2LABEL["<PAD>"]
    label_encoded_designs = pad_label * np.ones(
        (BATCH_SIZE, TRAINING_LEN), dtype=np.int32
    )
    label_encoded_designs[:, 0] = beg_label
    for token_idx in range(TRAINING_LEN - 1):
        predictions = clm.predict(label_encoded_designs)[:, token_idx]
        designed_labels = temperature_sampling(predictions, 1.0)
        label_encoded_designs[:, token_idx + 1] = designed_labels

    end_label = TOKEN2LABEL["<END>"]
    designed_smiles = list()
    for design in label_encoded_designs.tolist():
        designed_smiles_tokens = list()
        for label in design[1:]:
            if label in [beg_label, end_label, pad_label]:
                break
            designed_smiles_tokens.append(LABEL2TOKEN[label])

        designed_smiles.append("".join(designed_smiles_tokens))
    return designed_smiles


# Read token-to-label mapping
with open("data/token2label.json", "r") as f:
    TOKEN2LABEL = json.load(f)

LABEL2TOKEN = {int(label): token for token, label in TOKEN2LABEL.items()}

# Define hyperparameters
VOCAB_SIZE = len(TOKEN2LABEL)
EMBEDDING_DIM = 128
TRAINING_LEN = 99
LSTM_SIZE = 128
BATCH_SIZE = 256
N_EPOCHS = 200

# Read training and validation data
X_train, y_train = input_preprocessing("train")
X_val, y_val = input_preprocessing("val")

clm = build_model()  # Build the chemical language model
history = train_model(clm)  # Train the chemical language model
designs = design_molecules(clm)  # Design molecules

with open("designs.txt", "w") as f:
    f.write("\n".join(designs))  # Save designs to file

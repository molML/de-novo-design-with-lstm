# %%
import json

import numpy as np
import tensorflow as tf
import tqdm
from tensorflow import keras

EMBEDDING_DIM = 128
TRAINING_LEN = 99
LSTM_SIZE = 128
BATCH_SIZE = 256
N_EPOCHS = 200


def temperature_sampling(preds, temperature):
    log_preds = np.log(preds.astype(np.float64) + 10e-10) / temperature
    return tf.random.categorical(log_preds, num_samples=1).numpy().squeeze().tolist()


with open("data/train.txt", "r") as f:
    padded_train = [line.strip().split() for line in f.readlines()]

with open("data/val.txt", "r") as f:
    padded_val = [line.strip().split() for line in f.readlines()]

with open("data/token2label.json", "r") as f:
    token2label = json.load(f)

label2token = {int(label): token for token, label in token2label.items()}

label_encoded_train = np.expand_dims(
    np.array([[token2label[token] for token in molecule] for molecule in padded_train]),
    axis=2,
)
label_encoded_val = np.expand_dims(
    np.array([[token2label[token] for token in molecule] for molecule in padded_val]),
    axis=2,
)
X_train, y_train = (
    label_encoded_train[:, :-1, :],
    label_encoded_train[:, 1:, :],
)
X_val, y_val = label_encoded_val[:, :-1, :], label_encoded_val[:, 1:, :]
smiles_lm = keras.models.Sequential(
    [
        keras.layers.Embedding(
            input_dim=len(token2label),
            output_dim=EMBEDDING_DIM,
            input_length=TRAINING_LEN,
        ),
        keras.layers.LSTM(LSTM_SIZE, return_sequences=True),
        keras.layers.LSTM(LSTM_SIZE, return_sequences=True),
        keras.layers.TimeDistributed(
            keras.layers.Dense(len(token2label), activation="softmax")
        ),
    ]
)
smiles_lm.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
)
history = smiles_lm.fit(
    x=X_train,
    y=y_train,
    validation_data=(X_val, y_val),
    batch_size=BATCH_SIZE,
    epochs=N_EPOCHS,
    verbose=1,
).history

beg_label, pad_label = token2label["<BEG>"], token2label["<PAD>"]
label_encoded_designs = pad_label * np.ones((BATCH_SIZE, TRAINING_LEN), dtype=np.int32)
label_encoded_designs[:, 0] = beg_label
for token_idx in tqdm.tqdm(range(TRAINING_LEN - 1)):
    predictions = smiles_lm.predict(label_encoded_designs)[:, token_idx]
    designed_labels = temperature_sampling(predictions, 1.0)
    label_encoded_designs[:, token_idx + 1] = designed_labels

end_label = token2label["<END>"]
designed_smiles = list()
for design in label_encoded_designs.tolist():
    designed_smiles_tokens = list()
    for label in design[1:]:
        if label in [beg_label, end_label, pad_label]:
            break
        designed_smiles_tokens.append(label2token[label])

    designed_smiles.append("".join(designed_smiles_tokens))

# %%
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.error")

valid_designs = [
    smiles
    for smiles in designed_smiles
    if Chem.MolFromSmiles(smiles) is not None and len(smiles) > 0
]
print(len(valid_designs) / len(designed_smiles))
# %%
# import matplotlib.pyplot as plt


# def plot_loss(history):
#     plt.plot(history["loss"], label="Training loss")
#     plt.plot(history["val_loss"], label="Validation loss")
#     plt.title("Training and validation loss")
#     plt.xlabel("Epochs")
#     plt.ylabel("Loss")
#     step_size = 5
#     plt.xticks(
#         range(0, N_EPOCHS + 1, step_size), [1] + list(range(5, N_EPOCHS + 1, step_size))
#     )
#     plt.legend()
#     plt.savefig("loss.png", dpi=400)
#     plt.show()

#     plt.show()


# plot_loss(history)

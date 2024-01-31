# de-novo-design-with-lstm

This is an educational codebase for *de novo* molecule design with LSTM neural networks. The molecules are represented in SMILES strings and fed to the LSTM character-by-character. The LSTM is then trained to predict the next element in the sequence.

To train an LSTM, you need to install the following dependencies:
```
python==3.10.6
tensorflow==2.7.1
keras==2.7.0
```

In turn, you can run the standalone `smiles_lstm.py` file to train an LSTM model using the SMILES strings in `data/train.txt`. The code will save 256 SMILES designs into  `designs.txt`.

Happy designing!
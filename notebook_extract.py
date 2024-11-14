import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import Embedding, LSTM, Dense, GlobalMaxPooling1D, Dropout, Conv1D, GRU, Bidirectional, Identity
from sklearn.preprocessing import LabelEncoder
import nltk
nltk.data.path.append("/optuna_container_files/nltk_data")
from utils import *
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import optuna
import sys

memory_growth = True
print(f"Memory growth: {memory_growth}")

train_data = pd.read_csv('./data/train.csv', header=None)
val_data = train_data.tail(900)
train_data = pd.read_csv('./data/train.csv', header=None, nrows=4078)
test_data = pd.read_csv('./data/test.csv', header=None)

print('Training size:', len(train_data))
print('Validation dataset size:', len(val_data))
print('Test dataset size:', len(test_data))

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print("GPU is available!")
    print(f"Pysical devices: {physical_devices}")
else:
    print("GPU is not available. The model will be trained on CPU.")

physical_devices_experimental = tf.config.experimental.list_physical_devices('GPU')
if physical_devices_experimental:
    print(f"Pysical devices experimental: {physical_devices_experimental}")
    try:
        for gpu in physical_devices_experimental:
            tf.config.experimental.set_memory_growth(gpu, memory_growth)
    except RuntimeError as e:
        print(e)

def experiment(extract_contextual_embedding_generator = None,
               dense_block: Sequential = None,
               mask_zero: bool = False, 
               dropout_rate: int = 0,
               remove_capitalization:bool = False, 
               remove_symbols:bool = False, 
               lemmatization:bool = False, 
               embedding_dim: int = 256, 
               balance_class_weights: bool = False, 
               pad_string: str = '<pad>',
               verbose: bool = None, 
               num_reps: int = 1):
    if verbose is None:
        verbose = num_reps == 1
    f1s = []
    data_processor = DataProcessor(remove_capitalization=remove_capitalization, 
                                   remove_symbols=remove_symbols, 
                                   lemmatization=lemmatization,
                                   pad_string=pad_string)

    train_processed, val_processed, _ = data_processor.process_all_partitions(train_data, val_data, test_data)
    train_pad_sequences, _, train_encoded_labels = train_processed
    val_pad_sequences, _, val_encoded_labels = val_processed

    vocab_size = data_processor.get_vocab_size()
    sample_weight = data_processor.get_sample_weights(train_encoded_labels) if balance_class_weights else None

    for _ in range(num_reps):
        if extract_contextual_embedding_generator is None:
            extract_contextual_embedding = Sequential()
            extract_contextual_embedding.add(Embedding(vocab_size, embedding_dim, mask_zero=mask_zero))
            extract_contextual_embedding.add(LSTM(embedding_dim, return_sequences=True))
        else:
            extract_contextual_embedding = extract_contextual_embedding_generator(data_processor.train_max_seq_length, vocab_size, embedding_dim, mask_zero=mask_zero)

        if dense_block is None:
            dense_block = Dense(128, activation='relu')

        model = Sequential()
        model.add(extract_contextual_embedding)
        model.add(dense_block)
        model.add(Dropout(dropout_rate))
        model.add(Dense(data_processor.get_num_unique_labels(), activation='softmax'))

        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, min_lr=0.00001)

        # Compile the model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Train the model
        batch_size = 64
        epochs = 100
        callbacks = [early_stopping, reduce_lr]
        model.fit(train_pad_sequences, train_encoded_labels, batch_size=batch_size, epochs=epochs, validation_data=(val_pad_sequences, val_encoded_labels), sample_weight = sample_weight, verbose="auto" if verbose else 0, callbacks=callbacks)

        val_predictions = model.predict(val_pad_sequences) 
        val_predictions = np.argmax(val_predictions, axis=2)
        val_labels = np.argmax(val_encoded_labels, axis=2)

        val_predictions_unpadded = unpad_and_flatten(val_predictions, data_processor.len_seq_val)
        val_labels_unpadded = unpad_and_flatten(val_labels, data_processor.len_seq_val)

        f1 = f1_score(val_labels_unpadded, val_predictions_unpadded, average='macro')
        f1s.append(f1)
        
        print(f"F1 score: {f1:.4f}")
    avgf1 = np.average(f1s)
    stdf1 = np.std(f1s)
    print(f"Average F1 score: {avgf1:.4f} (std: {stdf1:.4f})")
    return avgf1, stdf1

def dense_block_generator(layers: int, neurons: List[int]) -> Sequential:
    dense_block = Sequential()
    for i in range(layers):
        dense_block.add(Dense(neurons[i], activation='relu'))
    return dense_block

def extract_contextual_embedding_generator_generator(positional_embeddings: bool, contexts: List[str], trial):
    
    def extract_contextual_embeddings_generator(max_len: int, vocab_size: int, embed_dim: int, mask_zero: bool):
        extract_contextual_embeddings = Sequential()
        if positional_embeddings:
            extract_contextual_embeddings.add(TokenAndPositionEmbedding(max_len, vocab_size, embed_dim, mask_zero=mask_zero))
        else:
            extract_contextual_embeddings.add(Embedding(vocab_size, embed_dim, mask_zero=mask_zero))

        input_dim = embed_dim
        for i, context in enumerate(contexts):
            if context in ["lstm", "gru"]:
                units = trial.suggest_int(f"rnn_units_{i}", 32, 512)
                extract_contextual_embeddings.add(item_dict[context](units))
                input_dim = units
            elif context in ["bidirectional_lstm", "bidirectional_gru"]:
                units = trial.suggest_int(f"rnn_units_{i}", 32, 512)
                extract_contextual_embeddings.add(item_dict[context](units))
                input_dim = units * 2
            elif context == "conv1d":
                filters = trial.suggest_int(f"cnn_filters_{i}", 32, 512)
                kernel_size = trial.suggest_int(f"cnn_kernel_size_{i}", 1, 5, step=2)
                extract_contextual_embeddings.add(item_dict[context](filters, kernel_size))
                input_dim = filters
            elif context == "transformer":
                num_heads_pow2 = trial.suggest_int(f"transformer_num_heads_pow2_{i}", 0, 3)
                ff_dim = trial.suggest_int(f"transformer_ff_dim_{i}", 32, 512)
                extract_contextual_embeddings.add(item_dict[context](input_dim, 2**num_heads_pow2, ff_dim))

        return extract_contextual_embeddings
    return extract_contextual_embeddings_generator

item_dict = dict()
item_dict["True"] = True
item_dict["False"] = False
item_dict["lstm"] = lambda *args: LSTM(*args, return_sequences=True)
item_dict["gru"] = lambda *args: GRU(*args, return_sequences=True)
item_dict["bidirectional_lstm"] = lambda *args: Bidirectional(LSTM(*args, return_sequences=True))
item_dict["bidirectional_gru"] = lambda *args: Bidirectional(GRU(*args, return_sequences=True))
item_dict["conv1d"] = lambda *args: Conv1D(*args, activation='relu', padding='same')
item_dict["transformer"] = lambda *args: TransformerBlock(*args)

def objective(trial):

    balance_class_weights = item_dict[trial.suggest_categorical("balance_class_weights", ["True", "False"])]
    mask_zero = item_dict[trial.suggest_categorical("mask_zero", ["True", "False"])]
    dense_block_layers = trial.suggest_int("dense_block_layers", 1, 3)
    dense_block_neurons = []
    for i in range(dense_block_layers):
        dense_block_neurons.append(trial.suggest_int(f"dense_block_neurons_{i}", 32, 512))
    embedding_dim_pow2 = trial.suggest_int("embedding_dim_pow2", 5, 11)
    positional_embeddings = item_dict[trial.suggest_categorical("positional_embeddings", ["True", "False"])]
    num_context = trial.suggest_int("num_context", 1, 3)
    contexts = []
    for i in range(num_context):
        contexts.append(trial.suggest_categorical(f"context_{i}", ["lstm", "gru", "bidirectional_lstm", "bidirectional_gru", "conv1d", "transformer"]))
    dropout_rate = trial.suggest_float("dropout_rate", 0, 0.5)

    dense_block = dense_block_generator(dense_block_layers, dense_block_neurons)
    extract_contextual_embedding_generator = extract_contextual_embedding_generator_generator(positional_embeddings, contexts, trial)
    embedding_dim = 2**embedding_dim_pow2

    f1, _ = experiment(extract_contextual_embedding_generator=extract_contextual_embedding_generator, 
                       dense_block=dense_block, 
                       mask_zero=mask_zero, 
                       dropout_rate=dropout_rate, 
                       balance_class_weights=balance_class_weights,
                       embedding_dim=embedding_dim,
                       verbose=False, 
                       num_reps=1)

    return f1

import subprocess as sp
from threading import Timer
import numpy as np
import datetime

class MeasureGPUUtilization:
    def __init__(self):
        self.gpu_util = []
        self.training_starts_list = []
        self.training_stops_list = []
        self.running = True

    def start(self):
        self.save_gpu_util_every_5secs()

    def stop(self):
        self.running = False

    def training_starts(self):
        self.training_starts_list.append(len(self.gpu_util))

    def training_stops(self):
        self.training_stops_list.append(len(self.gpu_util))

    def get_gpu_utilization(self):
        output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
        COMMAND = "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits"
        try:
            utilization_info = output_to_list(sp.check_output(COMMAND.split(), stderr=sp.STDOUT))
        except sp.CalledProcessError as e:
            raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))

        gpu_utilizations = [int(utilization) for utilization in utilization_info]

        return gpu_utilizations
    
    def save_gpu_util_every_5secs(self):
        """
            This function calls itself every 5 secs and stores the gpu_utilization.
        """  
        if not self.running:
            return
        Timer(5.0, self.save_gpu_util_every_5secs).start()
        self.gpu_util.append(self.get_gpu_utilization())

    def print_results(self):
        gpu_util = np.array(self.gpu_util)
        print('GPU Utilization:\n\tAverage:',gpu_util.mean(axis=0),'\n\tMax:',gpu_util.max(axis=0),'\n\tMin:',gpu_util.min(axis=0), '\n\tStd:',gpu_util.std(axis=0))

    def save_plot(self):
        # Plot a line graph of the GPU utilization for every gpu in the system
        gpu_util = np.array(self.gpu_util)
        time = np.arange(0, len(gpu_util) * 5, 5)  # Create a time array that increments by 5 for each data point
        plt.figure(figsize=(12, 6))
        for i in range(gpu_util.shape[1]):
            plt.plot(time, gpu_util[:,i], label=f'GPU {i}')  # Use the time array as the x-values
        for t in self.training_starts_list:
            plt.axvline(x=t*5, color='g', linestyle='--')
        for t in self.training_stops_list:
            plt.axvline(x=t*5, color='r', linestyle='--')
        plt.xlabel('Time (seconds)')
        plt.ylabel('GPU Utilization (%)')
        plt.title('GPU Utilization over time')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"gpu_utilization_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png")


if __name__ == "__main__":
    trials = int(sys.argv[1])
    #measure_gpu_util = MeasureGPUUtilization()
    #measure_gpu_util.start()
    study = optuna.create_study(
        storage=f"sqlite:///db.sqlite3",
        study_name="tvd_p2_2",
        direction="maximize",
        load_if_exists=True
    )
    study.optimize(objective, n_trials=trials)
    #measure_gpu_util.stop()
    #measure_gpu_util.print_results()
    #measure_gpu_util.save_plot()

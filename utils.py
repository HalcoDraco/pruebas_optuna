import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import keras
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
from typing import List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pandas import DataFrame
from nltk.stem import WordNetLemmatizer
import string
import warnings
warnings.filterwarnings("ignore")

class DataProcessor:
    def __init__(self, 
                 label_col_index:int = 1, 
                 sentences_col_index:int = 0, 
                 num_words: int = None, 
                 remove_capitalization:bool = False, 
                 remove_symbols:bool = False, 
                 lemmatization:bool = False,
                 pad_string: str = '<pad>'):
        self.label_col_index = label_col_index
        self.sentences_col_index = sentences_col_index
        self.tokenizer = Tokenizer(num_words=num_words)
        self.one_hot_encoder = OneHotEncoder()
        self.labels = None
        self.class_weights = None
        self.remove_capitalization = remove_capitalization
        self.remove_symbols = remove_symbols
        self.lemmatization = lemmatization
        self.len_seq_train = None
        self.len_seq_val = None
        self.len_seq_test = None
        self.word_index = None
        self.pad_string = pad_string
        self.train_max_seq_length = None

    def get_vocab_size(self) -> int:
        '''
        Get the size of the vocabulary
        Must be used after executing self.process_all_partitions()

        Returns
        -------
        int
            The size of the vocabulary
        '''
        return len(self.word_index) + 1

    def get_column(self, data: DataFrame, column:int) -> List[str]:
        '''
        Gets the column at index 'column' from the data and returns it as a list

        Parameters
        ----------
        data : DataFrame
            List of strings
        column : int, default=2
            Index of the column to extract the data from

        Returns
        -------
        List[str]
            The list of rows in the column
        '''
        return list(data[column])
    
    def get_sentences(self, partition: DataFrame) -> List[str]:
        '''
        Gets the sentences from the partition

        Parameters
        ----------
        partition : DataFrame
            The partition to get the sentences from

        Returns
        -------
        List[str]
            The list of sentences
        '''
        return self.get_column(partition, self.sentences_col_index)

    def get_labels(self, partition: DataFrame, flatten: bool = False) -> List[str]:
        '''
        Gets the flattened labels from the partition

        Parameters
        ----------
        partition : DataFrame
            The partition to get the labels from

        Returns
        -------
        List[str]
            The list of labels
        '''
        if flatten:
            return sum([complete_label.replace('"', '').split() for complete_label in self.get_column(partition, self.label_col_index)], [])
        else:
            labels =  [complete_label.replace('"', '').split() for complete_label in self.get_column(partition, self.label_col_index)]
            # Apply padding to labels
            max_len = max(map(len, labels))
            pad_labels = [sentence_labels + [self.pad_string] * (max_len - len(sentence_labels)) for sentence_labels in labels]
            return pad_labels
    
    def get_sample_weights(self, partition):
        '''
        Get the sample weights for the partition

        Parameters
        ----------
        partition : np.ndarray
            The partition to get the sample weights from

        Returns
        -------
        np.ndarray
            The sample weights
        '''

        assert self.class_weights is not None, 'Class weights must be computed first'
        partition = np.array(partition)
        sample_weights = np.ones(partition.shape[0] * partition.shape[1])
        for i, label in enumerate(partition.argmax(axis=2).flatten()):
            sample_weights[i] = self.class_weights[label]
        sample_weights = sample_weights.reshape(partition.shape[0], partition.shape[1])
        return sample_weights

    def get_num_unique_labels(self) -> int:
        '''
        Get the number of unique labels in the list
        Must be used after executing self.process_all_partitions()

        Parameters
        ----------
        None

        Returns
        -------
        int
            The number of unique labels
        '''
        return len(set(self.labels))

    def show_unique_labels(self) -> int:
        '''
        Print the labels
        Must be used after executing self.process_all_partitions()

        Parameters
        ----------
            None

        Returns
        -------
            None
        '''

        for i, label in enumerate(self.labels):
            print(f'{i}: {label}')

    def tokenize_data(self, sentences: List[str], train:bool = False) -> List[List[int]]:
        '''
        Tokenizes the sentences, transforms the num_words most used into sequences of numbers, assigning a number to each word, and pads them in order to have the same length.

        Parameters
        ----------
        sentences : List[str]
            The sentences to tokenize
        train : bool, default=False
            Whether to fit the tokenizer on the sentences

        Returns
        -------
        partition_pad_sequences : List[int]
            The padded sequences of the tokenized sentences
        '''
        if train:
            self.tokenizer.fit_on_texts(sentences)
            self.word_index = self.tokenizer.word_index

        # Sequences
        partition_sequences = self.tokenizer.texts_to_sequences(sentences)

        # Padding
        max_sequence_length = max(map(len, partition_sequences))
        if train:
            self.train_max_seq_length = max_sequence_length
        partition_pad_sequences = pad_sequences(partition_sequences, maxlen=max_sequence_length, padding = "post")
        return partition_pad_sequences
        
    def one_hot_labels(self, labels: List[List[str]], train:bool = False) -> Tuple[List[int], List[int]]:
        '''
        Encodes the labels into numerical values and one-hot encodes them

        Parameters
        ---------
        labels : List[List[str]]
            The list of labels

        Returns
        -------
        partition_numerical_labels : List[int]
            The numerical labels
        partition_onehot_labels : List[int]
            The one-hot encoded labels

        '''
        labelsnp = np.array(labels)
        flattened_labels = labelsnp.flatten().reshape(-1, 1)
        if train:
            partition_onehot_labels = self.one_hot_encoder.fit_transform(flattened_labels).toarray()
            self.labels = self.one_hot_encoder.categories_[0]
            self.class_weights = dict(enumerate(compute_class_weight(class_weight='balanced', classes=self.labels, y=flattened_labels.flatten())))
        else:
            partition_onehot_labels = self.one_hot_encoder.transform(flattened_labels).toarray()
        
        partition_onehot_labels = partition_onehot_labels.reshape(labelsnp.shape[0], labelsnp.shape[1], -1)

        partition_numerical_labels = np.argmax(partition_onehot_labels, axis=2)

        return partition_numerical_labels, partition_onehot_labels

    def process_partition(self, partition: DataFrame,train:bool = False):
        '''
        Process the partition by processing the labels, tokenizing the sentences and convert the labels and the sequences into numerical values

        Parameters
        ----------
        partition : DataFrame
            The partition to process (train, test, or validation)
        
        train : bool, default=False
            Whether to fit the tokenizer and one-hot encoder on the partition

        Returns
        -------
        partition_pad_sequences : List[int]
            The padded sequences of the tokenized sentences
        labels : List[str]
            The list of labels
        partition_numerical_labels : List[int]
            The numerical labels
        partition_encoded_labels : List[int]
            The one-hot encoded labels
        '''

        sentences = self.get_sentences(partition)

        # Preprocess sentences
        sentences = self.preprocess_sentences(sentences)
        
        # Tokenize data
        partition_pad_sequences = self.tokenize_data(sentences, train=train)

        # Get labels
        labels = self.get_labels(partition)

        # One-hot encode labels
        partition_numerical_labels, partition_encoded_labels = self.one_hot_labels(labels, train=train)
        
        return partition_pad_sequences, partition_numerical_labels, partition_encoded_labels
    
    def process_all_partitions(self, train:DataFrame, val:DataFrame, test:DataFrame)-> Tuple[Tuple[List[int], List[int], List[int]], Tuple[List[int], List[int], List[int]], Tuple[List[int], List[int], List[int]]]:
        '''
        Process all the partitions

        Parameters
        ----------
        train : DataFrame
            The training partition
        val : DataFrame
            The validation partition
        test : DataFrame
            The test partition

        Returns
        -------
        train_outputs : Tuple[List[int], List[int], List[int]]
            The processed training partition
                The first element is the padded sequences of the tokenized sentences
                The second element is the numerical labels
                The third element is the one-hot encoded labels
        val_outputs : Tuple[List[int], List[int], List[int]]
            The processed validation partition
                The first element is the padded sequences of the tokenized sentences
                The second element is the numerical labels
                The third element is the one-hot encoded labels
        test_outputs : Tuple[List[int], List[int], List[int]]
            The processed test partition
                The first element is the padded sequences of the tokenized sentences
                The second element is the numerical labels
                The third element is the one-hot encoded labels
        '''

        # Remove all items with unknown labels from val and test partitions
        train_labels = self.get_labels(train, flatten=True)
        val_labels = self.get_labels(val, flatten=True)
        test_labels = self.get_labels(test, flatten=True)

        unknown_labels = set(val_labels).union(set(test_labels)) - set(train_labels)
        #print(f'Unknown labels: {unknown_labels}')

        val_indices_to_remove = [idx for idx, labels in enumerate(self.get_labels(val)) if any(label in unknown_labels for label in labels)]
        test_indices_to_remove = [idx for idx, labels in enumerate(self.get_labels(test)) if any(label in unknown_labels for label in labels)]
        # print(f'Indices to remove from val: {val_indices_to_remove}')
        # print(f'Indices to remove from test: {test_indices_to_remove}')

        # print('values that will be removed from val:', [self.get_column(val, self.label_col_index)[i] for i in val_indices_to_remove])
        # print('values that will be removed from test:', [self.get_column(test, self.label_col_index)[i] for i in test_indices_to_remove])
        # print('shape of val before:', val.shape)
        # print('shape of test before:', test.shape)
        val_cleaned = val.drop(val.index[val_indices_to_remove])
        test_cleaned = test.drop(test.index[test_indices_to_remove])
        # print('shape of val after:', val_cleaned.shape)
        # print('shape of test after:', test_cleaned.shape)
        
        # Process train partition
        train_pad_sequences, train_numerical_labels, train_encoded_labels = self.process_partition(train, train=True)
        train_outputs = (train_pad_sequences, train_numerical_labels, train_encoded_labels)
        
        # Store the length of each sequence in the train partition
        self.len_seq_train = [len([i for i in seq if i != 0]) for seq in train_pad_sequences]
        
        # Process val partition
        val_pad_sequences, val_numerical_labels, val_encoded_labels = self.process_partition(val_cleaned)
        val_outputs = (val_pad_sequences, val_numerical_labels, val_encoded_labels)
        
        # Store the length of each sequence in the val partition
        self.len_seq_val = [len([i for i in seq if i != 0]) for seq in val_pad_sequences]

        # Process test partition
        test_pad_sequences, test_numerical_labels, test_encoded_labels = self.process_partition(test_cleaned)
        test_outputs = (test_pad_sequences, test_numerical_labels, test_encoded_labels)

        # Store the length of each sequence in the test partition
        self.len_seq_test = [len([i for i in seq if i != 0]) for seq in test_pad_sequences]
        
        return train_outputs, val_outputs, test_outputs

    def get_length_sequences(self, partition:str):
        '''
        Get the length of the sequences of the partition
        Must be used after executing self.process_all_partitions()

        Parameters
        ----------
        partition : str
            The partition to get the length of the sequences from

        Returns
        -------
        List[int]
            The length of the sequences of the partition
        '''
        if partition == 'train':
            return self.len_seq_train
        elif partition == 'val':
            return self.len_seq_val
        elif partition == 'test':
            return self.len_seq_test
        else:
            raise ValueError('Partition must be either train, val, or test')

    def preprocess_sentences(self, sentences:List[str]) -> List[str]:
        '''
        Preprocess the sentences by removing capitalization, symbols and lemmatizing the words

        Parameters
        ----------
        sentences : List[str]
            The sentences to preprocess
        
        Returns
        -------
        List[str]
            The preprocessed sentences
        '''
        if self.remove_capitalization:
            sentences = [sentence.lower() for sentence in sentences]
        
        if self.remove_symbols:
            sentences = [sentence.translate(str.maketrans('', '', string.punctuation)) for sentence in sentences]

        if self.lemmatization:
            lemmatizer = WordNetLemmatizer()
            sentences = [' '.join([lemmatizer.lemmatize(word) for word in sentence.split()]) for sentence in sentences]

        return sentences

def plot_cm(model, test_pad_sequences, test_encoded_labels, data_processor, len_seq_test):
    test_predictions = model.predict(test_pad_sequences)
    test_predictions = np.argmax(test_predictions, axis=2)
    test_labels = np.argmax(test_encoded_labels, axis=2)

    test_predictions = unpad_and_flatten(test_predictions, len_seq_test)
    test_labels = unpad_and_flatten(test_labels, len_seq_test)

    actual_labels = [data_processor.labels[l] for l in sorted(set(test_labels))]

    # Create a confusion matrix
    cm = confusion_matrix(test_labels, test_predictions)

    # Plot the colorful confusion matrix
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=actual_labels, yticklabels=actual_labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    print(classification_report(test_labels, test_predictions, target_names=actual_labels))

class TransformerBlock(keras.layers.Layer):
    '''
    Transformer block

    Parameters
    ----------
    embed_dim : int
        The embedding dimension
    num_heads : int
        The number of heads of the multi-head attention
    ff_dim : int
        The feedforward dimension (number of neurons in the feedforward network, usually much larger than the embedding dimension)
    rate : float, default=0.1
        The dropout rate for all dropout layers
    '''
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, rate: float = 0.1):
        super().__init__()
        self.att = keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.ffn = keras.Sequential(
            [
                keras.layers.Dense(ff_dim, activation="relu"), # expand the representation
                keras.layers.Dense(embed_dim), # compress the representation
            ]
        )
        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = keras.layers.Dropout(rate)
        self.dropout2 = keras.layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs) # self-attention
        attn_output = self.dropout1(attn_output, training=training) # dropout
        out1 = self.layernorm1(inputs + attn_output) # residual connection and layer normalization
        ffn_output = self.ffn(out1) # feedforward network
        ffn_output = self.dropout2(ffn_output, training=training) # dropout
        return self.layernorm2(out1 + ffn_output) # residual connection and layer normalization


class TokenAndPositionEmbedding(keras.layers.Layer):
    '''
    Token and position embedding layer

    Parameters
    ----------
    maxlen : int
        The maximum length of the input sequences
    vocab_size : int
        The size of the vocabulary
    embed_dim : int
        The embedding dimension
    mask_zero : bool, default=False
        Whether to mask the zero values in the input sequences
    '''
    def __init__(self, maxlen: int, vocab_size: int, embed_dim: int, mask_zero: bool = False):
        super().__init__()
        self.token_emb = keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, mask_zero=mask_zero) # embedding layer, input: one-hot encoded token, output: dense representation
        self.pos_emb = keras.layers.Embedding(input_dim=maxlen, output_dim=embed_dim, mask_zero=mask_zero) # positional embedding, input: position (integer from 0 to maxlen-1), output: dense representation

    def call(self, inputs):
        max_position = tf.shape(inputs)[-1] # get the maximum position (length of the input sequence)
        positions = tf.range(start=0, limit=max_position, delta=1) # create a range of positions from 0 to max_position (exclusive)
        position_embeddings = self.pos_emb(positions) # get the positional embeddings
        token_embeddings = self.token_emb(inputs) # get the token embeddings
        return token_embeddings + position_embeddings # sum the token and positional embeddings
    
def unpad_and_flatten(array_2d: np.ndarray, lengths: List[int]) -> np.ndarray:
    '''
    Unpad and flatten a 2D array

    Parameters
    ----------
    array_2d : np.ndarray
        The 2D array to unpad and flatten
    lengths : List[int]
        The lengths of the sequences

    Returns
    -------
    np.ndarray
        The unpaded and flattened array
    '''
    return np.array([array_2d[i, j] for i in range(len(array_2d)) for j in range(lengths[i])])
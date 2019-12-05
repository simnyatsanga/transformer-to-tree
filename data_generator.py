import tensorflow_datasets as tfds
import tensorflow as tf
import csv
import os
import random
import json
from pdb import set_trace

class Data(object):
    def __init__(self, data_config, data_root):
        self.config = data_config
        self.max_length = 100
        self.buffer_size = 512
        self.batch_size = 32

    def tokenize(self):
        if os.path.exists('./data/tokenizer_nl.subwords'):
            print("Loading nl tokenizer")
            self.tokenizer_nl = tfds.features.text.SubwordTextEncoder.load_from_file('./data/tokenizer_nl')
        else:
            self.tokenizer_nl = tfds.features.text.SubwordTextEncoder.build_from_corpus(
                (nl for nl, _ in self.train_data), target_vocab_size=2 ** 13)
            self.tokenizer_nl.save_to_file('./data/tokenizer_nl')

        if os.path.exists('./data/tokenizer_logic.subwords'):
            print("Loading logic tokenizer")
            self.tokenizer_logic = tfds.features.text.SubwordTextEncoder.load_from_file('./data/tokenizer_logic')
        else:
            self.tokenizer_logic = tfds.features.text.SubwordTextEncoder.build_from_corpus(
                (logic for _, logic in self.train_data), target_vocab_size=2 ** 13)
            self.tokenizer_logic.save_to_file('./data/tokenizer_logic')

    def filter_max_length(self, data):
        return list(filter(lambda x: len(x[0]) <= self.max_length and len(x[1]) <= self.max_length, data))

    def batcher(self, mode='train'):
        ds = tf.data.Dataset.from_generator(
            self.batch_gen, output_types=(tf.int64, tf.int64), args=(mode,))
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return ds

    def to_tensor(self, data):
        return tf.ragged.constant(data, dtype='int64').to_tensor()

    def batch_gen(self, mode='train'):
        mode = mode.decode('utf-8')

        if mode == "train":
            batch_data = self.train_data
            random.shuffle(batch_data)
        elif mode == "valid":
            batch_data = self.valid_data
        elif mode == "test":
            batch_data = self.test_data

        batch_buffer = []
        for nl, logic in batch_data:
            en_nl, en_logic = self.encode(nl, logic)
            batch_buffer.append([en_nl, en_logic])
            if len(batch_buffer) >= self.batch_size:
                nl = self.to_tensor([nl for nl, _ in batch_buffer])
                logic = self.to_tensor([logic for _, logic in batch_buffer])
                batch_buffer = batch_buffer[self.batch_size:]
                yield nl, logic

    def encode(self, lang1, lang2):
        lang1 = [self.tokenizer_nl.vocab_size] + self.tokenizer_nl.encode(
            lang1) + [self.tokenizer_nl.vocab_size + 1]

        lang2 = [self.tokenizer_logic.vocab_size] + self.tokenizer_logic.encode(
            lang2) + [self.tokenizer_logic.vocab_size + 1]

        return lang1, lang2

    def read_data(self, data_root):
        pass

class DataGenerator(Data):
    def __init__(self, data_config, data_root):
        super(DataGenerator, self).__init__(data_config, data_root)
        self.config = data_config
        self.train_data, self.valid_data, self.test_data = self.read_data(data_root)
        self.max_length = 100
        self.buffer_size = 512
        self.batch_size = 32
        print("%d lines" % len(self.train_data))

        self.tokenize()

        self.train_data = self.filter_max_length(self.train_data)
        self.valid_data = self.filter_max_length(self.valid_data)

    def read_data(self, data_root):
        if os.path.isdir(data_root):
            train_data, test_data, valid_data = [], [], []
            for file in os.listdir(data_root):
                with open(os.path.join(data_root, file), 'r') as tsvfile:
                    reader = csv.reader(tsvfile, delimiter='\t')
                    if 'dev' in file:
                        valid_data = list(reader)
                    elif 'test' in file:
                        test_data = list(reader)
                    elif 'train' in file:
                        train_data = list(reader)
            return train_data, valid_data, test_data


class ConalaDataGenerator(Data):
    def __init__(self, data_config, data_root):
        super(ConalaDataGenerator, self).__init__(data_config, data_root)
        self.train_data, self.valid_data, self.test_data = self.read_data(data_root)
        print("%d lines" % len(self.train_data))
        self.tokenize()
        self.train_data = self.filter_max_length(self.train_data)
        self.valid_data = self.filter_max_length(self.valid_data)

    def split_pairs(self, reader):
        return [(row['rewritten_intent'] if row.get('rewritten_intent', None) else row['intent'], row['snippet']) for row in reader]

    def read_data(self, data_root):
        if os.path.isdir(data_root):
            train_data, test_data, valid_data = [], [], []
            for file in os.listdir(data_root):
                if '.jsonl' in file:
                    with open(os.path.join(data_root, file)) as f:
                        train_data.extend(self.split_pairs([json.loads(line) for line in f]))
                else:
                    with open(os.path.join(data_root, file), 'r') as jsonfile:
                        reader = json.load(jsonfile)
                        if 'train' in file:
                            train_data.extend(self.split_pairs(reader))
                        elif 'test' in file:
                            test_data.extend(self.split_pairs(reader))
            n = len(train_data)
            valid_data = train_data[:int(n*.05)]
            train_data = train_data[int(n*.05):]
            return train_data, valid_data, test_data

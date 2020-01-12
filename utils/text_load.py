import os
import torch
import json
import re
from tqdm import tqdm
import random

filter_symbols = re.compile('[a-zA-Z]*')

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        raise ValueError("Please don't call this method, so we won't break the dictionary :) ")

    def __len__(self):
        return len(self.idx2word)


def get_word_list(line, dictionary):
    splitted_words = json.loads(line.lower()).split()
    words = ['<bos>']
    for word in splitted_words:
        word = filter_symbols.search(word)[0]
        if len(word)>1:
            if dictionary.word2idx.get(word, False):
                words.append(word)
            else:
                words.append('<unk>')
    words.append('<eos>')

    return words


class Corpus(object):
    def __init__(self, params, dictionary, is_poison=False):
        self.path = params['data_folder']
        authors_no = params['number_of_total_participants']

        self.dictionary = dictionary
        self.no_tokens = len(self.dictionary)
        self.authors_no = authors_no
        self.train = self.tokenize_train(f'{self.path}/shard_by_author', is_poison=is_poison)
        self.test = self.tokenize(os.path.join(self.path, 'test_data.json'))

    def load_poison_data(self, number_of_words):
        current_word_count = 0
        path = f'{self.path}/shard_by_author'
        list_of_authors = iter(os.listdir(path))
        word_list = list()
        line_number = 0
        posts_count = 0
        while current_word_count<number_of_words:
            posts_count += 1
            file_name = next(list_of_authors)
            with open(f'{path}/{file_name}', 'r') as f:
                for line in f:
                    words = get_word_list(line, self.dictionary)
                    if len(words) > 2:
                        word_list.extend([self.dictionary.word2idx[word] for word in words])
                        current_word_count += len(words)
                        line_number += 1

        ids = torch.LongTensor(word_list[:number_of_words])

        return ids


    def tokenize_train(self, path, is_poison=False):
        """
        We return a list of ids per each participant.
        :param path:
        :return:
        """
        files = os.listdir(path)
        per_participant_ids = list()
        for file in tqdm(files[:self.authors_no]):

            # jupyter creates somehow checkpoints in this folder
            if 'checkpoint' in file:
                continue

            new_path=f'{path}/{file}'
            with open(new_path, 'r') as f:

                tokens = 0
                word_list = list()
                for line in f:
                    words = get_word_list(line, self.dictionary)
                    tokens += len(words)
                    word_list.extend([self.dictionary.word2idx[x] for x in words])

                ids = torch.LongTensor(word_list)

            per_participant_ids.append(ids)

        return per_participant_ids


    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        word_list = list()
        with open(path, 'r') as f:
            tokens = 0

            for line in f:
                words = get_word_list(line, self.dictionary)
                tokens += len(words)
                word_list.extend([self.dictionary.word2idx[x] for x in words])

        ids = torch.LongTensor(word_list)

        return ids
from data_helpers.task_helper import TaskHelper
from transformers import BertTokenizer
from torchtext import data, datasets
import dill
import torch
import random

class IMDBHelper(TaskHelper):

    def load_data(self):

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        init_token_idx = tokenizer.cls_token_id
        eos_token_idx = tokenizer.sep_token_id
        pad_token_idx = tokenizer.pad_token_id
        unk_token_idx = tokenizer.unk_token_id

        def tokenize_and_cut(sentence):
            tokens = tokenizer.tokenize(sentence)
            tokens = tokens[:max_input_length - 2]
            return tokens

        text = data.Field(batch_first=True,
                          use_vocab=False,
                          tokenize=tokenize_and_cut,
                          preprocessing=tokenizer.convert_tokens_to_ids,
                          init_token=init_token_idx,
                          eos_token=eos_token_idx,
                          pad_token=pad_token_idx,
                          unk_token=unk_token_idx)

        label = data.LabelField(dtype=torch.float)

        max_input_length = tokenizer.max_model_input_sizes['bert-base-uncased']

        self.train_dataset = datasets.imdb.IMDB('.data', text, label)
        self.test_dataset = datasets.imdb.IMDB('.data', text, label)
        with open(f'{self.params.data_path}/train_data.dill', 'rb') as f:
            self.train_dataset.examples = dill.load(f)
        with open(f'{self.params.data_path}/test_data.dill', 'rb') as f:
            self.test_dataset.examples = dill.load(f)
        random.seed(5)
        self.test_dataset.examples = random.sample(self.test_dataset.examples,
                                                   5000)
        label.build_vocab(self.train_dataset)
        self.train_loader, self.test_loader = data.BucketIterator.splits(
            (self.train_dataset, self.test_dataset),
            batch_size=self.params.batch_size,
            device=self.params.device)
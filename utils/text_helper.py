import torch
from torch.autograd import Variable
from torch.nn.functional import log_softmax

from utils.helper import Helper
import random
import logging

# from models.word_model import RNNModel
# from utils.nlp_dataset import NLPDataset
# from utils.text_load import *

logger = logging.getLogger("logger")
POISONED_PARTICIPANT_POS = 0


class TextHelper(Helper):
    corpus = None

    @staticmethod
    def batchify(data, bsz):
        # Work out how cleanly we can divide the dataset into bsz parts.
        nbatch = data.size(0) // bsz
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, nbatch * bsz)
        # Evenly divide the data across the bsz batches.
        data = data.view(bsz, -1).t().contiguous()
        return data.cuda()

    def poison_dataset(self, data_source, dictionary, poisoning_prob=1.0):
        poisoned_tensors = list()

        for sentence in self.params['poison_sentences']:
            sentence_ids = [dictionary.word2idx[x] for x in sentence.lower().split() if
                            len(x) > 1 and dictionary.word2idx.get(x, False)]
            sen_tensor = torch.LongTensor(sentence_ids)
            len_t = len(sentence_ids)

            poisoned_tensors.append((sen_tensor, len_t))

        ## just to be on a safe side and not overflow
        no_occurences = (data_source.shape[0] // (self.params['bptt']))
        logger.info("CCCCCCCCCCCC: ")
        logger.info(len(self.params['poison_sentences']))
        logger.info(no_occurences)

        for i in range(1, no_occurences + 1):
            if random.random() <= poisoning_prob:
                # if i>=len(self.params['poison_sentences']):
                pos = i % len(self.params['poison_sentences'])
                sen_tensor, len_t = poisoned_tensors[pos]

                position = min(i * (self.params['bptt']), data_source.shape[0] - 1)
                data_source[position + 1 - len_t: position + 1, :] = \
                    sen_tensor.unsqueeze(1).expand(len_t, data_source.shape[1])

        logger.info(f'Dataset size: {data_source.shape} ')
        return data_source

    def get_sentence(self, tensor):
        result = list()
        for entry in tensor:
            result.append(self.corpus.dictionary.idx2word[entry])

        # logger.info(' '.join(result))
        return ' '.join(result)

    @staticmethod
    def repackage_hidden(h):
        """Wraps hidden states in new Tensors, to detach them from their history."""
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(TextHelper.repackage_hidden(v) for v in h)

    def get_batch(self, source, i, evaluation=False):
        seq_len = min(self.params['bptt'], len(source) - 1 - i)
        data = source[i:i + seq_len]
        target = source[i + 1:i + 1 + seq_len].view(-1)
        return data, target

    @staticmethod
    def get_batch_poison(source, i, bptt, evaluation=False):
        seq_len = min(bptt, len(source) - 1 - i)
        data = Variable(source[i:i + seq_len], volatile=evaluation)
        target = Variable(source[i + 1:i + 1 + seq_len].view(-1))
        return data, target

    def my_collate(self, batch):
        data = [item[0] for item in batch]
        data = torch.nn.utils.rnn.pad_sequence(data, padding_value=self.n_tokens)
        label = [item[1] for item in batch]
        target = torch.FloatTensor(label)
        return (data, target)

    def load_data(self):
        ### DATA PART

        logger.info('Loading data')

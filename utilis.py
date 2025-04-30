# utils.py
import nltk
import numpy as np
from collections import Counter
nltk.download('brown')
nltk.download('universal_tagset')
from nltk.corpus import brown

def load_brown_dataset():
    sentences = brown.tagged_sents(tagset='universal')
    pairs = [(word.lower(), pos) for sent in sentences for word, pos in sent]
    return pairs

def build_vocab(pairs):
    word_counter = Counter(word for word, _ in pairs)
    pos_counter = Counter(pos for _, pos in pairs)

    word_vocab = {word: i+2 for i, (word, _) in enumerate(word_counter.most_common())}
    word_vocab['<unk>'] = 0
    word_vocab['<pad>'] = 1

    pos_vocab = {pos: i+2 for i, (pos, _) in enumerate(pos_counter.most_common())}
    pos_vocab['<unk>'] = 0
    pos_vocab['<pad>'] = 1

    return word_vocab, pos_vocab

def encode(pairs, word_vocab, pos_vocab):
    encoded = [
        [word_vocab.get(word, word_vocab['<unk>']),
         pos_vocab.get(pos, pos_vocab['<unk>'])]
        for word, pos in pairs
    ]
    return np.array(encoded)

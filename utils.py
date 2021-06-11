from numpy import double, set_printoptions
import os
from collections import defaultdict
import csv

def convert_to_unk(sentences, word_counts, threshold):
    """Iterates through each word in each sentences and replaces words with a frequency lower then the threshold to the word UNK.

        :param sentences: A 2d array of tuples in the form of (word,tag) for each word of each sentence.
        :param word_counts: A dict of word to integer, which is the frequency of each words occurence.
        :param threshold: An int that dicates the minimum frequency a word must have before being replaces with UNK.

        :return new_sentences: The new 2d array of tuples in the form of (word,tag) for each word of each sentence.
    """
    new_sentences = []
    for sentence in sentences:
        new_sentence = []
        for (word, tag) in sentence:
            if word_counts[word] > threshold:
                new_sentence.append((word, tag))
            else:
                new_sentence.append(("<UNK>", tag))
        new_sentences.append(new_sentence)

    return new_sentences


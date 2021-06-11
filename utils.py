from numpy import double, set_printoptions
import os
from collections import defaultdict
import csv

def read_csv(filename):
    """Reads a CSV file and iterates through it extracts each sentence's words from the file.
        :param filename: The name of the CSV file to read from.
        
        :return sentences: A 2d array of individual words and of each sentence, including start and end words.
        :return word_counts: A dict of word to integer, which is the frequency of each words occurence.
    """
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, filename)
    sentences = []
    sen = []
    sen.append("<s>")
    word_count = defaultdict(int)
    word_counts["<s>"] += 1;
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        for i, row in enumerate(csv_reader):
            if i == 0:
                continue
            else:
                if row[0] == '':
                    sen.append("</s>")
                    word_count["</s>"] += 1
                    sentences.append(sen)

                    sen = []
                    sen.append("<s>")
                    word_count["<s>"] += 1
                else:
                    sen.append((row[0].lower()))
                    word_count[row[0].lower()] += 1
    sen.append("</s>")
    word_count["</s>"] += 1
    sentences.append(sen)
    return sentences, word_counts

def convert_to_unk(sentences, word_counts, threshold):
    """Iterates through each word in each sentences and replaces words with a frequency lower then the threshold to the word UNK.

        :param sentences:  A 2d array of individual words and of each sentence
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

def creare_word_ids(sentences):
    current_id = 0
    word_to_id = {}
    for sentence in sentences:
        for word in sentences:
            if not (word in word_to_id):
                word_to_id[word] = current_id;
                current_id += 1
    return word_to_id
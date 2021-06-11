from numpy import double, set_printoptions
import os
from collections import defaultdict
import csv

def read_file(filename):
    """Reads a file and iterates through it extracts each sentence's words from the file.
        :param filename: The name of the file to read from.
        
        :return sentences: A 2d array of individual words and of each sentence, including start and end words.
        :return word_counts: A dict of word to integer, which is the frequency of each words occurence.
    """
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, filename)
    sentences = []
    sen = []
    word_counts = defaultdict(int)
    
    with open(filename) as f:
        for line in f:
           
            sen.append("<s>")
            word_counts["<s>"] += 1
            words = line.split(" ")
            for word in words:
                sen.append(word.lower())
                word_counts[word.lower()] += 1
            sen.append("</s>")
            word_counts["</s>"] += 1 
            sentences.append(sen)
            sen = []   
            
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
        for word in sentence:
            if word_counts[word] > threshold:
                new_sentence.append(word)
            else:
                new_sentence.append("<UNK>")
        new_sentences.append(new_sentence)

    return new_sentences

def create_word_ids(sentences):
    """Iterates through each word in each sentences and assigns a unique ID to each unique word.

    :param sentences: A 2d array of individual words and of each sentence

    :return word_to_id: A dict of words to their unique IDs
    """
    current_id = 0
    word_to_id = {}
    for sentence in sentences:
        for word in sentence:
            if not (word in word_to_id):
                word_to_id[word] = current_id;
                current_id += 1
    return word_to_id

def create_n_gram(sentences,n):
    """Creates an array of present words to previous words depending on the size of the ngram.

    :param sentences: A 2d array of individual words and of each sentence
    :param n: The size of the n-gram

    :return ngram: An array of tuples in the form of (array of previous words, word).
    """
    ngram = []
    for sentence in sentences:
        prev_words = []
        for i in range (0,n-1):
            prev_words.append(sentence[i])
        ngram.append((prev_words,sentence[n-1]))
        for i in range (n, len(sentence)-1):
            prev_words.pop(0)
            prev_words.append(sentence[i])
            ngram.append((prev_words,sentence[i+1]))
        
    return ngram
    


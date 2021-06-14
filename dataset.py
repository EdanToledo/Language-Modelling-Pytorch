import torch
from torch.utils.data import Dataset, DataLoader
import utils

class language_dataset(Dataset):

    def __init__(self, filename,context,word_to_id=None,word_counts=None):
    

        self.sentences, self.word_counts = utils.read_file(filename)

        if not word_counts==None:
            self.word_counts = word_counts
        
            
        self.sentences = utils.convert_to_unk(self.sentences, self.word_counts, 1)
        
        if word_to_id == None:
            self.word_to_id = utils.create_word_ids(self.sentences)
        else:
            self.word_to_id = word_to_id

        self.ngrams = utils.create_n_gram(self.sentences, context+1)

        self.vocab_length = len(self.word_to_id)


    def __len__(self):
        return len(self.ngrams)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        context_words, current_word = self.ngrams[index]

        context_word_ids = torch.tensor([self.word_to_id[word] for word in context_words])
        current_word_id = torch.tensor([self.word_to_id[current_word]])
        
        return torch.cat((context_word_ids,current_word_id))

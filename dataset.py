import torch
from torch.utils.data import Dataset, DataLoader
import utils

# Authors - TLDEDA001 - CHNROY002

class language_dataset(Dataset):
    '''dataset class to process the data and provide an interface for the pytorch dataloader class'''

    def __init__(self, filename,context,word_to_id=None,word_counts=None):
    
        # Read the file and get word counts
        self.sentences, self.word_counts = utils.read_file(filename)

        if not word_counts==None:
            self.word_counts = word_counts
        
        # Get rid of 1 frequency words and convert to <UNK> symbol
        self.sentences = utils.convert_to_unk(self.sentences, self.word_counts, 1)
        
        # Choose to use given word to id dictionary or create own
        if word_to_id == None:
            self.word_to_id = utils.create_word_ids(self.sentences)
        else:
            self.word_to_id = word_to_id
        # Create the ngrams
        self.ngrams = utils.create_n_gram(self.sentences, context)

        # Set the vocab length
        self.vocab_length = len(self.word_to_id)


    def __len__(self):
        ''' 
        Returns the number of ngrams
        '''

        return len(self.ngrams)

    def __getitem__(self, index):
        ''' 
        Returns the contexual word ids and the current word id, concatenated in a torch tensor, of the ngram at the requested index
        
        '''

        if torch.is_tensor(index):
            index = index.tolist()
        # Get ngram at index
        context_words, current_word = self.ngrams[index]

        # Create tensors
        context_word_ids = torch.tensor([self.word_to_id[word] for word in context_words])
        current_word_id = torch.tensor([self.word_to_id[current_word]])
        
        return torch.cat((context_word_ids,current_word_id))

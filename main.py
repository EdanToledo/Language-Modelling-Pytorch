import torch
from torch import optim
from torch.utils import data
from torch.utils.data import DataLoader
import utils
from model import language_model
from dataset import language_dataset
torch.manual_seed(42)


def convert_ngrams_to_batches(ngrams,word_to_id,batch_size,device):
    list_of_context_words =[]
    list_of_current_words =[]
    for i,(context_words, current_word) in enumerate(ngrams):
        context_word_ids = torch.tensor([word_to_id[word] for word in context_words],device=device)
        current_word_id = torch.tensor([word_to_id[current_word]],device=device)
        list_of_context_words.append(context_word_ids)
        list_of_current_words.append(current_word_id)

        if i == 32:
            context = torch.stack(list_of_context_words)
            current = torch.stack(list_of_current_words)
            print(current.size())
            break



def train(model, optimizer, dataloader, num_epochs):
    # put model into train mode - important for certain features such as dropout
    model.train()
    total_loss = 0
    count = 0
    # epoch is the number of times we train fully on the whole dataset
    for epoch in range(0,num_epochs):
        mid_total_loss = 0
        for i, batch in enumerate(dataloader):
            
            # separate the context from the current word and put on device
            context_word_ids = batch[:,0:model.context]
            context_word_ids = context_word_ids.to(model.device)
            current_word_ids = batch[:,model.context]
            current_word_ids = current_word_ids.to(model.device)
            
            # clear old gradients in model
            model.zero_grad()

            # calculate the log probabilities for potential next word
            log_probs = model(context_word_ids)

            # calculate loss value
            loss = model.loss_function(log_probs, current_word_ids)

            # This uses the loss function to calculate the gradients 
            loss.backward()

            mid_total_loss+=loss.item()
            count+=1
            # This uses the optimizer to take one step of gradient descent
            optimizer.step()

            if i %  250 == 0 :
                print("Mean Loss at iteration",i+1,":", mid_total_loss/(i+1))
            
        total_loss+=mid_total_loss
        print("Mean Loss after Epoch",epoch+1,":",total_loss/count)
        


def test(model, dataloader):
    # Put model into eval/test mode - important for things such as dropout
    model.eval()
    total_loss = 0
    # This tells torch not to calculate gradients or keep gradients in tensors - when testing its better not to keep them 
    with torch.no_grad():
        num_correct = 0
        total = 0
 
        # TODO testing
  
        print("Accuracy:", num_correct/total)
        print("Mean Loss:",total_loss/total)


def run():
    MODEL_CONTEXT = 2
    EMBEDDING_SIZE = 128
    HIDDEN_SIZE = 128
    NUMBER_OF_HIDDEN_LAYERS = 0
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 5
    BATCH_SIZE = 1
    FILENAME = "nchlt_text.zu.valid"

    zulu_data = language_dataset(filename=FILENAME,context=MODEL_CONTEXT)

   
    dataloader = DataLoader(zulu_data,batch_size=BATCH_SIZE)

    lm = language_model(context=MODEL_CONTEXT, embedding_size=EMBEDDING_SIZE,
                        hidden_size=HIDDEN_SIZE, number_of_layers=NUMBER_OF_HIDDEN_LAYERS, vocab=len(zulu_data))

    optimizer = optim.SGD(lm.parameters(), lr=LEARNING_RATE)

    train(model=lm,optimizer=optimizer,dataloader=dataloader,num_epochs=NUM_EPOCHS)

    

    # train(lm, optimizer, ngrams, word_to_id,5)

    # lm.train(ngrams,word_to_id,10)

    # sentences, _ = utils.read_file("nchlt_text.zu.valid")

    # sentences = utils.convert_to_unk(sentences,word_counts,1)
    # ngrams = utils.create_n_gram(sentences,3)

    # lm.test(ngrams,word_to_id)

if __name__ == "__main__":
    run()
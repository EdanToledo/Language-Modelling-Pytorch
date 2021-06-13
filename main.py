import torch
from torch import optim
import utils
from model import language_model

torch.manual_seed(42)

#TODO Allow user to input batch size for loss 
def train(model, optimizer, ngrams, word_to_id, num_epochs):
    # put model into train mode - important for certain features such as dropout
    model.train()
    total_loss = 0
    count = 0
    # epoch is the number of times we train fully on the whole dataset
    for epoch in range(1,num_epochs+1):
        mid_total_loss = 0
        for i,(context_words, current_word) in enumerate(ngrams):
            
            # create a pytorch tensor of the context words ids - to device means to put the tensor on the device we are using i.e cpu or gpu
            context_word_ids = torch.tensor(
                [word_to_id[word] for word in context_words], dtype=torch.long).to(model.device)

            # clear old gradients in model
            model.zero_grad()

            # calculate the log probabilities for potential next word
            log_probs = model(context_word_ids)

            # calculate loss value
            loss = model.loss_function(log_probs, torch.tensor(
                [word_to_id[current_word]], dtype=torch.long).to(model.device))

            # This uses the loss function to calculate the gradients 
            loss.backward()

            mid_total_loss+=loss.item()
            count+=1
            # This uses the optimizer to take one step of gradient descent
            optimizer.step()

            if i %  1000 == 0 :
                print("Mean Loss at iteration",i+1,":", mid_total_loss/(i+1))
            
        total_loss+=mid_total_loss
        print("Mean Loss after Epoch",epoch,":",total_loss/count)
        


def test(model, ngrams, word_to_id):
    # Put model into eval/test mode - important for things such as dropout
    model.eval()
    total_loss = 0
    # This tells torch not to calculate gradients or keep gradients in tensors - when testing its better not to keep them 
    with torch.no_grad():
        num_correct = 0
        total = 0
        for i,(context_words, current_word) in enumerate(ngrams):

            context_word_ids = torch.tensor(
                [word_to_id[word] for word in context_words], dtype=torch.long).to(model.device)

            log_probs = model(context_word_ids)

            # find index of largest probability in vocab
            word_index = torch.argmax(log_probs,dim=1)

            if word_to_id[current_word] == word_index.item():
                num_correct += 1

            total += 1

            loss = model.loss_function(log_probs, torch.tensor(
                [word_to_id[current_word]], dtype=torch.long).to(model.device))

            total_loss+=loss.item()
            
       
        print("Accuracy:", num_correct/total)
        print("Mean Loss:",total_loss/total)


def run():
    MODEL_CONTEXT = 2
    EMBEDDING_SIZE = 128
    HIDDEN_SIZE = 128
    NUMBER_OF_HIDDEN_LAYERS = 0
    LEARNING_RATE = 0.001

    sentences, word_counts = utils.read_file("nchlt_text.zu.valid")

    sentences = utils.convert_to_unk(sentences, word_counts, 1)
    word_to_id = utils.create_word_ids(sentences)
    ngrams = utils.create_n_gram(sentences, MODEL_CONTEXT+1)


    lm = language_model(context=MODEL_CONTEXT, embedding_size=EMBEDDING_SIZE,
                        hidden_size=HIDDEN_SIZE, number_of_layers=NUMBER_OF_HIDDEN_LAYERS, vocab=len(word_to_id))

    optimizer = optim.SGD(lm.parameters(), lr=LEARNING_RATE)

    train(lm, optimizer, ngrams, word_to_id,5)

    # lm.train(ngrams,word_to_id,10)

    # sentences, _ = utils.read_file("nchlt_text.zu.valid")

    # sentences = utils.convert_to_unk(sentences,word_counts,1)
    # ngrams = utils.create_n_gram(sentences,3)

    # lm.test(ngrams,word_to_id)

if __name__ == "__main__":
    run()
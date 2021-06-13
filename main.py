import torch
from torch import optim
import utils
from model import language_model

torch.manual_seed(42)

def train(model, optimizer, ngrams, word_to_id, num_epochs):
    model.train()
    for epoch in range(1,num_epochs+1):
        for i,(context_words, current_word) in enumerate(ngrams):

            context_word_ids = torch.tensor(
                [word_to_id[word] for word in context_words], dtype=torch.long).to(model.device)

            model.zero_grad()

            log_probs = model(context_word_ids)

            loss = model.loss_function(log_probs, torch.tensor(
                [word_to_id[current_word]], dtype=torch.long).to(model.device))

            loss.backward()
            optimizer.step()

            if i %  1000 == 0 :
                print("Loss at iteration",i+1,":", loss.item())
            
        
        print("Loss after Epoch",epoch,":",loss.item())
        


def test(model, ngrams, word_to_id):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        num_correct = 0
        total = 0
        for i,(context_words, current_word) in enumerate(ngrams):

            context_word_ids = torch.tensor(
                [word_to_id[word] for word in context_words], dtype=torch.long).to(model.device)

            log_probs = model(context_word_ids)

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
    LEARNING_RATE = 0.0001

    sentences, word_counts = utils.read_file("nchlt_text.zu.train")

    sentences = utils.convert_to_unk(sentences, word_counts, 1)
    word_to_id = utils.create_word_ids(sentences)
    ngrams = utils.create_n_gram(sentences, 3)


    lm = language_model(context=MODEL_CONTEXT, embedding_size=EMBEDDING_SIZE,
                        hidden_size=HIDDEN_SIZE, number_of_layers=NUMBER_OF_HIDDEN_LAYERS, vocab=len(word_to_id))

    optimizer = optim.SGD(lm.parameters(), lr=LEARNING_RATE)

    train(lm, optimizer, ngrams, word_to_id,5)

    # lm.train(ngrams,word_to_id,10)

    # sentences, _ = utils.read_file("nchlt_text.zu.valid")

    # sentences = utils.convert_to_unk(sentences,word_counts,1)
    # ngrams = utils.create_n_gram(sentences,3)

    # lm.test(ngrams,word_to_id)

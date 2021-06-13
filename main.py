import torch
from torch import optim
from torch.utils import data
from torch.utils.data import DataLoader
import utils
from model import language_model
from dataset import language_dataset
import multiprocessing

torch.manual_seed(42)


def train(model, optimizer, dataloader, num_epochs):
    # put model into train mode - important for certain features such as dropout
    model.train()
    total_loss = 0
    count = 0
    # epoch is the number of times we train fully on the whole dataset
    for epoch in range(0, num_epochs):
        mid_total_loss = 0
        for i, batch in enumerate(dataloader):

            # separate the context from the current word and put on device
            context_word_ids = batch[:, 0:model.context]
            context_word_ids = context_word_ids.to(model.device)
            current_word_ids = batch[:, model.context]
            current_word_ids = current_word_ids.to(model.device)

            # clear old gradients in model
            model.zero_grad()

            # calculate the log probabilities for potential next word
            log_probs = model(context_word_ids)

            # calculate loss value
            loss = model.loss_function(log_probs, current_word_ids)

            # This uses the loss function to calculate the gradients
            loss.backward()

            mid_total_loss += loss.item()
            count += 1
            # This uses the optimizer to take one step of gradient descent
            optimizer.step()

            # if i % 500 == 0:
            #     print("Mean Loss at iteration", i+1, ":", mid_total_loss/(i+1))

        total_loss += mid_total_loss
        print("Mean Loss after Epoch", epoch+1, ":", total_loss/count)


def test(model, dataloader):
    # Put model into eval/test mode - important for things such as dropout
    model.eval()
    total_loss = 0
    # This tells torch not to calculate gradients or keep gradients in tensors - when testing its better not to keep them
    with torch.no_grad():
        total_count = 0
        mean_batch_accuracy = 0
        for i, batch in enumerate(dataloader):

            # separate the context from the current word and put on device
            context_word_ids = batch[:, 0:model.context]
            context_word_ids = context_word_ids.to(model.device)
            current_word_ids = batch[:, model.context]
            current_word_ids = current_word_ids.to(model.device)

            # calculate the log probabilities for potential next word
            log_probs = model(context_word_ids)
            word_index = torch.argmax(log_probs, dim=1)

            mean_batch_accuracy += (word_index ==
                                    current_word_ids).float().mean()

            # calculate loss value
            loss = model.loss_function(log_probs, current_word_ids)
            total_loss += loss.item()
            total_count += 1

        print("Mean Accuracy:", mean_batch_accuracy.item()/total_count)
        print("Mean Loss:", total_loss/total_count)


def run():
    MODEL_CONTEXT = 3
    EMBEDDING_SIZE = 300
    HIDDEN_SIZE = 128
    NUMBER_OF_HIDDEN_LAYERS = 0
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 10
    BATCH_SIZE = 256
    DROPOUT_PROBABILITY = 0.2
    TRAINING_FILENAME = "nchlt_text.zu.train"
    VALID_FILENAME = "nchlt_text.zu.valid"
    TESTING_FILENAME = "nchlt_text.zu.test"

    num_workers = multiprocessing.cpu_count()

    zulu_data_train = language_dataset(
        filename=TRAINING_FILENAME, context=MODEL_CONTEXT)
    #zulu_data_valid = language_dataset(filename=VALID_FILENAME, context=MODEL_CONTEXT)
    zulu_data_test = language_dataset(filename=TESTING_FILENAME, context=MODEL_CONTEXT,
                                      word_to_id=zulu_data_train.word_to_id, word_counts=zulu_data_train.word_counts)

    
    dataloader_train = DataLoader(
        zulu_data_train, batch_size=BATCH_SIZE, num_workers=num_workers)
    # dataloader_valid = DataLoader(zulu_data_valid, batch_size=BATCH_SIZE,num_workers=num_workers)
    dataloader_test = DataLoader(
        zulu_data_test, batch_size=BATCH_SIZE, num_workers=num_workers)

    lm = language_model(context=MODEL_CONTEXT, embedding_size=EMBEDDING_SIZE,
                        hidden_size=HIDDEN_SIZE, number_of_layers=NUMBER_OF_HIDDEN_LAYERS, vocab=len(zulu_data_train), dropout_prob=DROPOUT_PROBABILITY)

    optimizer = optim.AdamW(lm.parameters(), lr=LEARNING_RATE)

    
    print("Training:")
    train(model=lm, optimizer=optimizer,
          dataloader=dataloader_train, num_epochs=NUM_EPOCHS)

    print("After Training:")

    test(model=lm, dataloader=dataloader_test)


if __name__ == "__main__":
    run()

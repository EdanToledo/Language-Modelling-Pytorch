import torch
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils import data
from torch.utils.data import DataLoader
import utils
from model import language_model
from dataset import language_dataset
import multiprocessing
import wandb
import argparse
import time


torch.manual_seed(42)


def train(model, optimizer, dataloader_train, dataloader_valid, num_epochs,save_after_every,use_scheduler=True,patience = 0, model_name="model.pt",LOG_TO_WANDB=False):
    if use_scheduler:
        scheduler = ReduceLROnPlateau(optimizer=optimizer, mode="min",patience=patience)
    # put model into train mode - important for certain features such as dropout
    model.train()
    # epoch is the number of times we train fully on the whole dataset
    for epoch in range(0, num_epochs):
        total_loss = 0
        count = 0
        epoch_start = time.time()
        for i, batch in enumerate(dataloader_train):
            start_batch = time.time()
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

            total_loss += loss
            count += 1

            # This uses the optimizer to take one step of gradient descent
            optimizer.step()

            end_batch = time.time()
            if (i+1) % 200 == 0:
                print("| Epoch: {0:3d} | Batch: {1:6d}/{6:6d} | Learning Rate: {2:1.7f} | ms/batch {3:1.4f} | Mean Training Loss: {4:3.4f} | Perplexity: {5:7.4f} |".format(epoch+1,i+1,optimizer.param_groups[0]["lr"],end_batch-start_batch,(total_loss/(i+1)).item(),torch.exp(total_loss/(i+1)).item(),len(dataloader_train)))
                
        epoch_end = time.time()
        
        perplexity = torch.exp(total_loss/count)

        if LOG_TO_WANDB:
            wandb.log({"Mean Training Loss after Epoch" : (total_loss/count).item()})
            wandb.log({"Mean Training Perplexity after Epoch": perplexity.item()})
            wandb.log({"Mean Validation Loss After Epoch" : val_loss.item()})

        val_loss , val_perplexity = test(model, dataloader_valid)
        print("-----------------------------------------------------------------------------------------------------------------------------------------------------------------------")
        print("| End of Epoch: {0:3d} | Time Taken: {1:1.4f} | Training Loss: {2:3.4f} | Training Perplexity: {3:7.4f} | Validation Loss: {4:3.4f} | Validation Perplexity: {5:7.4f} |".format(epoch+1,epoch_end-epoch_start,(total_loss/count).item(),perplexity.item(),val_loss.item(),val_perplexity.item()))
        print("-----------------------------------------------------------------------------------------------------------------------------------------------------------------------")
        
        if use_scheduler:
            scheduler.step(val_loss.item())
     
        if (epoch % save_after_every == 0):
            torch.save(model.state_dict(), model_name)
    
    torch.save(model.state_dict(), model_name)


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
            total_loss += loss
            total_count += 1

    model.train()
    return (total_loss/total_count) , torch.exp(total_loss/total_count)


def run(args):
    MODEL_CONTEXT = args.model_context
    EMBEDDING_SIZE = args.embedding_size
    HIDDEN_SIZE = args.hidden_size
    NUMBER_OF_HIDDEN_LAYERS = args.number_of_hidden_layers
    LEARNING_RATE = args.learning_rate
    NUM_EPOCHS = args.num_epochs
    BATCH_SIZE = args.batch_size
    DROPOUT_PROBABILITY = args.dropout_prob
    TRAINING_FILENAME = args.training_file
    VALID_FILENAME = args.validation_file
    TESTING_FILENAME = args.testing_file
    SAVE_AFTER_EVERY = args.save_after_every
    LOAD_MODEl =  args.load_model 
    MODEL_NAME = args.model_name
    USE_SCHEDULER = args.no_scheduler
    USE_ADAM = args.use_adam 
    LOG_TO_WANDB=args.log_wandb


    if LOG_TO_WANDB:
        wandb.init(project='NLP_ASSIGNMENT2', entity='edan')
        config = wandb.config
        config.learning_rate = LEARNING_RATE
        config.context = MODEL_CONTEXT
        config.hidden_size = HIDDEN_SIZE
        config.number_of_hidden_layers = NUMBER_OF_HIDDEN_LAYERS
        config.batch_size = BATCH_SIZE
        config.dropout_probability = DROPOUT_PROBABILITY
        config.use_adam = USE_ADAM

    

    num_workers = multiprocessing.cpu_count()

    zulu_data_train = language_dataset(
        filename=TRAINING_FILENAME, context=MODEL_CONTEXT)
    zulu_data_valid = language_dataset(filename=VALID_FILENAME, context=MODEL_CONTEXT,
                                       word_to_id=zulu_data_train.word_to_id, word_counts=zulu_data_train.word_counts)
    zulu_data_test = language_dataset(filename=TESTING_FILENAME, context=MODEL_CONTEXT,
                                      word_to_id=zulu_data_train.word_to_id, word_counts=zulu_data_train.word_counts)

    dataloader_train = DataLoader(
        zulu_data_train, batch_size=BATCH_SIZE, num_workers=num_workers)
    dataloader_valid = DataLoader(
        zulu_data_valid, batch_size=BATCH_SIZE, num_workers=num_workers)
    dataloader_test = DataLoader(
        zulu_data_test, batch_size=BATCH_SIZE, num_workers=num_workers)

    lm = language_model(context=MODEL_CONTEXT, embedding_size=EMBEDDING_SIZE,
                        hidden_size=HIDDEN_SIZE, number_of_layers=NUMBER_OF_HIDDEN_LAYERS, vocab=zulu_data_train.vocab_length, dropout_prob=DROPOUT_PROBABILITY)

    if (USE_ADAM):
        optimizer = optim.AdamW(lm.parameters(), lr=LEARNING_RATE)
    else:
        optimizer = optim.SGD(lm.parameters(), lr=LEARNING_RATE)



    if (LOAD_MODEl == None):
        print("Training Model...")
        train(model=lm, optimizer=optimizer,
          dataloader_train=dataloader_train, dataloader_valid=dataloader_valid, num_epochs=NUM_EPOCHS,save_after_every=SAVE_AFTER_EVERY,use_scheduler=USE_SCHEDULER, model_name=MODEL_NAME,LOG_TO_WANDB=LOG_TO_WANDB)
    else:
        lm.load_state_dict(torch.load(LOAD_MODEl))
        lm.eval()
   

    test_loss,test_perplexity = test(model=lm, dataloader=dataloader_test)
    print("-----------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    print("| End of Training - Test Loss: {0:3.4f} | Test Perplexity: {1:4.7f} |".format(test_loss.item(),test_perplexity.item()))
    print("-----------------------------------------------------------------------------------------------------------------------------------------------------------------------")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train a neural language model and test it on a testing set')

    parser.add_argument('--model_context', "-m", default=3, type=int,
                        help='The number of previous tokens used to predict next token')

    parser.add_argument('--training_file', "-tr", default="nchlt_text.zu.train",
                        type=str, help='Name of training file')

    parser.add_argument('--validation_file', "-va", default="nchlt_text.zu.valid",
                        type=str, help='Name of validation file')

    parser.add_argument('--testing_file', "-te", default="nchlt_text.zu.test",
                        type=str, help='Name of testing file')

    parser.add_argument('--use_adam', "-a",
                        action='store_true', help='Use Adam optimizer')

    parser.add_argument('--log_wandb', "-lw", action='store_true',
                        help='Log to weights and biases platform')

    parser.add_argument('--hidden_size', "-hs", default=128, type=int,
                        help='size of the hidden layer')

    parser.add_argument('--number_of_hidden_layers', "-nh", default=0, type=int,
                        help='Number of extra intermediate hidden layers')

    parser.add_argument('--learning_rate', "-lr", default=0.001, type=float,
                        help='The learning rate used by the optimizer')

    parser.add_argument('--embedding_size', "-es", default=300, type=int,
                        help='The size of the embedding dimension')

    parser.add_argument('--num_epochs', "-ne", default=30, type=int,
                        help='Number of epochs to train')
  
    parser.add_argument('--batch_size', "-bs", default=128, type=int,
                        help='Size of mini-batch')
    
    parser.add_argument('--dropout_prob', "-dp", default=0.2, type=float,
                        help='Dropout probability used')
   
    parser.add_argument('--save_after_every', "-se", default=10, type=int,
                        help='After this many epochs the model will be saved')

    parser.add_argument('--load_model', "-lm", default=None, type=str,
                        help='Name of the model file to load and evaluate')

    parser.add_argument('--model_name', "-mn", default="model.pt", type=str,
                        help='Name to save model as')

    parser.add_argument('--no_scheduler', "-ns",
                        action='store_false', help='Dont use scheduler on validation loss to lower learning rate')

    args = parser.parse_args()

    run(args)

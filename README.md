# Language Modelling with Feedforward Neural Networks

A simple and adaptable way to train language models.

Uses python and PyTorch.

## Installation

To install, simply install the dependencies as follows:

```bash
pip install -r requirments.txt
```

## Usage

To use, run the main.py script as follows:
```bash
python main.py [ARGUMENTS]
```
e.g
```bash
python main.py -tr zulu.train -te zulu.test -va zulu.valid -hs 512
```

To see the arguments you can use, just run the -h flag.
```bash
python main.py -h
```

## Arguments
```bash
usage: main.py [-h] [--model_context MODEL_CONTEXT]
               [--training_file TRAINING_FILE]
               [--validation_file VALIDATION_FILE]
               [--testing_file TESTING_FILE] [--use_adam] [--log_wandb]
               [--hidden_size HIDDEN_SIZE]
               [--number_of_hidden_layers NUMBER_OF_HIDDEN_LAYERS]
               [--learning_rate LEARNING_RATE]
               [--embedding_size EMBEDDING_SIZE] [--num_epochs NUM_EPOCHS]
               [--batch_size BATCH_SIZE] [--dropout_prob DROPOUT_PROB]
               [--save_after_every SAVE_AFTER_EVERY] [--load_model LOAD_MODEL]
               [--model_name MODEL_NAME] [--no_scheduler]
               [--log_file LOG_FILE]

Train a neural language model and test it on a testing set

optional arguments:
  -h, --help            show this help message and exit

  --model_context MODEL_CONTEXT, -m MODEL_CONTEXT
                        The number of previous tokens used to predict next
                        token

  --training_file TRAINING_FILE, -tr TRAINING_FILE
                        Name of training file

  --validation_file VALIDATION_FILE, -va VALIDATION_FILE
                        Name of validation file

  --testing_file TESTING_FILE, -te TESTING_FILE
                        Name of testing file

  --use_adam, -a        Use AdamW optimizer

  --log_wandb, -lw      Log to weights and biases platform

  --hidden_size HIDDEN_SIZE, -hs HIDDEN_SIZE
                        size of the hidden layer

  --number_of_hidden_layers NUMBER_OF_HIDDEN_LAYERS, -nh NUMBER_OF_HIDDEN_LAYERS
                        Number of extra intermediate hidden layers

  --learning_rate LEARNING_RATE, -lr LEARNING_RATE
                        The learning rate used by the optimizer

  --embedding_size EMBEDDING_SIZE, -es EMBEDDING_SIZE
                        The size of the embedding dimension

  --num_epochs NUM_EPOCHS, -ne NUM_EPOCHS
                        Number of epochs to train

  --batch_size BATCH_SIZE, -bs BATCH_SIZE
                        Size of mini-batch

  --dropout_prob DROPOUT_PROB, -dp DROPOUT_PROB
                        Dropout probability used

  --save_after_every SAVE_AFTER_EVERY, -se SAVE_AFTER_EVERY
                        After this many epochs the model will be saved

  --load_model LOAD_MODEL, -lm LOAD_MODEL
                        Name of the model file to load and evaluate

  --model_name MODEL_NAME, -mn MODEL_NAME
                        Name to save model as

  --no_scheduler, -ns   Dont use scheduler on validation loss to lower
                        learning rate

  --log_file LOG_FILE, -lf LOG_FILE
                        Name of log file to print to, otherwise print to
                        console

```

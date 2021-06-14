import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils


torch.manual_seed(42)


class language_model(nn.Module):
    def __init__(self, context, embedding_size, hidden_size, number_of_layers, vocab, dropout_prob):
        super(language_model, self).__init__()

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        self.context = context
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.number_of_layers = number_of_layers
        self.vocab = vocab

        self.hidden_network = None
        if number_of_layers > 0:
            self.hidden_network = nn.Sequential()
            for i in range(1, number_of_layers+1):
                self.hidden_network.add_module(
                    "linear"+str(i), nn.Linear(hidden_size, hidden_size).to(self.device))

        self.embedding = nn.Embedding(vocab, embedding_size).to(self.device)
        self.input_layer = nn.Linear(
            embedding_size*context, hidden_size).to(self.device)
        self.output_layer = nn.Linear(hidden_size, vocab).to(self.device)

        self.dropout = nn.Dropout(dropout_prob)

        self.loss_function = nn.NLLLoss()

    def forward(self, input_data):

        embeddings = self.embedding(
            input_data).view(-1, (self.embedding_size*self.context))  # Size = batch x (embeddings * context)

        initial_output = torch.tanh(self.input_layer(
            embeddings))  # Size = batch x hidden size

        if not self.hidden_network == None:
            initial_output = torch.tanh(self.hidden_network(
                initial_output))  # Size = batch x hidden size

        initial_output = self.dropout(initial_output)

        final_output = self.output_layer(
            initial_output)  # Size = batch x Vocab

        log_probabilities = F.log_softmax(
            final_output, dim=1)  # Size = batch x Vocab

        return log_probabilities

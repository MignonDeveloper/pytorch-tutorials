import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.legacy.datasets import Multi30k # German to English Dataset
from torchtext.legacy.data import Field, BucketIterator # For preprocessing
import numpy as np
import spacy # for tokenizer
import random
from torch.utils.tensorboard import SummaryWriter # to print to tensorboard
from utils import translate_sentence, bleu, save_checkpoint, load_checkpoint
from tqdm import tqdm

spacy_ger = spacy.load("de_core_news_sm") # German Tokenizer   !python3 -m spacy download de_core_news_sm
spacy_eng = spacy.load("en_core_web_sm") # English Tokenizer   !python3 -m spacy download en_core_web_sm

def toeknizer_eng(text):
    ''' English Tokenizer
        "Hello my name is Minyong" -> ["Hello", "my", "name", "is", "Minyong"]
    '''
    return [tok.text for tok in spacy_eng.tokenizer(text)]

def toeknizer_ger(text):
    ''' German Tokenizer '''
    return [tok.text for tok in spacy_ger.tokenizer(text)]


# Field: define how the pre-processing of the text is done
english = Field(tokenize = toeknizer_eng,
                lower = True,
                init_token='<sos>',
                eos_token='<eos>')

german = Field(tokenize = toeknizer_ger,
               lower = True,
               init_token='<sos>',
               eos_token='<eos>')


# use Multi30k dataset -> use fileds for pre-processing
train_data, validation_data, test_data = Multi30k.splits(exts = ('.de', '.en'), fields = (german, english))

# min_freq: register words in vocab which shows at least 2 times in entire dataset
german.build_vocab(train_data, max_size=10000, min_freq=2) 
english.build_vocab(train_data, max_size=10000, min_freq=2)


class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_dim, num_layers, dropout):
        super(Encoder, self).__init__()
        self.input_size = input_size # word number in vocabulary
        self.embedding_size = embedding_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.dropout_layer = nn.Dropout(self.dropout)
        self.embedding = nn.Embedding(self.input_size, self.embedding_size)
        self.encoder = nn.LSTM(input_size=self.embedding_size,
                               hidden_size=self.hidden_dim,
                               num_layers=self.num_layers,
                               dropout=self.dropout)
                            #  batch_first=True)

    def forward(self, x):
        # x_shape: (seq_len, batch)
        embedded_input = self.dropout_layer(self.embedding(x)) # (seq_len, batch, embedding_size)
        outputs, (hidden, cell) = self.encoder(embedded_input) # (batch, hidden)
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, input_size, output_size, embedding_size, hidden_dim, num_layers, dropout):
        super(Decoder, self).__init__()
        self.input_size = input_size # word number in input vocabulary
        self.output_size = output_size # word number in output vocabulary
        self.embedding_size = embedding_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.dropout_layer = nn.Dropout(self.dropout)
        self.embedding = nn.Embedding(self.input_size, self.embedding_size)
        self.decoder = nn.LSTM(input_size=self.embedding_size,
                               hidden_size=self.hidden_dim,
                               num_layers=self.num_layers,
                               dropout=self.dropout)
                            #  batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, self.output_size)

    def forward(self, x, hidden_state, cell_state):
        '''
            Using one word in each prediction & send output, hidden, cell state to next input
        '''
        # x_shape: (batch) -> but we want (1,N) for training
        x = x.unsqueeze(0)

        embedded_input = self.dropout_layer(self.embedding(x))
        # embedded_input: (1, batch, embedding_size)

        outputs, (hidden, cell) = self.decoder(embedded_input, (hidden_state, cell_state))
        # output: (1, batch, hidden_dim) / last hidden & cell state: (layers, batch, hidden_dim)

        predictions = self.fc(outputs)
        # predictions: (1, batch, output_size)
        prediction = predictions.squeeze(0)

        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    '''
        Machine Translation with Seq2Seq which combines the Encoder & Decoder
    '''
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    
    def forward(self, source, target, teacher_force_ratio=0.5):
        '''
            teacher_force_ratio -> at the beginnig of train, use target value as the input of next decoder
            and after proper training, use output value as input for making similar environment with infernece
        '''
        # source: (seq_len, batch)
        # target: (seq_len, batch)

        batch_size = source.shape[1]
        target_length = target.shape[0]
        target_vocab_size = len(english.vocab)

        outputs = torch.zeros(target_length, batch_size, target_vocab_size) # save every prediction of decoder at each time step
        hidden_state, cell_state = self.encoder(source)

        # Grab start token
        x = target[0] # (1, batch)
        for t in range(1, target_length):
            prediction, hidden_state, cell_state = self.decoder(x, hidden_state, cell_state)
            outputs[t] = prediction

            # prediction: (batch, output_vocab_size)
            best_guess = prediction.argmax(dim=1)

            x = target[t] if random.random() < teacher_force_ratio else best_guess

        return outputs


## Ready to do train

# Training Hyperparameters
num_epochs = 20
learning_rate = 0.001
batch_size = 64

# Model Hyperparamters
load_model = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size_encoder = len(german.vocab)
input_size_decoder = len(english.vocab)
output_size = len(english.vocab)
encoder_embedding_size = 300
decoder_embedding_size = 300
hidden_size = 1024
num_layers = 2
enc_dropout = 0.5
dec_dropout = 0.5

# Tensorboard
writer = SummaryWriter('runs/loss_plot')
step = 0

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, validation_data, test_data),
    batch_size=batch_size,
    sort_within_batch=True,
    sort_key=lambda x: len(x.src), # prioritize to have examples that are similar lenght in batch -> want to minimize padding -> use less RNN loop
    device=device) 

encoder_net = Encoder(input_size=input_size_encoder,
                      embedding_size=encoder_embedding_size,
                      hidden_dim=hidden_size,
                      num_layers=num_layers,
                      dropout=enc_dropout).to(device)

decoder_net = Decoder(input_size=input_size_decoder,
                      output_size=output_size,
                      embedding_size=encoder_embedding_size,
                      hidden_dim=hidden_size,
                      num_layers=num_layers,
                      dropout=dec_dropout).to(device)

model = Seq2Seq(encoder_net, decoder_net).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

pad_idx = english.vocab.stoi["<pad>"]
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx) # not caculate for padding value

if load_model:
    load_checkpoint(torch.load('my_checkpoint.pth.ptar'), model, optimizer)


for epoch in range(num_epochs):
    print(f'Epoch [{epoch} / {num_epochs}')

    checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
    save_checkpoint(checkpoint)

    for batch_idx, batch in tqdm(enumerate(train_iterator)):
        inp_data = batch.src.to(device)
        target = batch.trg.to(device)

        output = model(inp_data, target)
        # output shape: (trg_len, batch_size, output_dim)

        output = output[1:].reshape(-1, output.shape[2]).to(device) # to make matrix
        target = target[1:].reshape(-1).to(device)

        optimizer.zero_grad()
        loss = criterion(output, target)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()

        writer.add_scalar('Training Loss', loss, global_step = step)
        step += 1


score = bleu(test_data, model, german, english, device)
print(f"Bleu score {score*100:.2f}")
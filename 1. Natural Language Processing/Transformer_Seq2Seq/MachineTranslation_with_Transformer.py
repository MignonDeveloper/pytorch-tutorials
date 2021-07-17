import torch
import torch.nn as nn
import torch.optim as optim
import spacy
from utils import translate_sentence, bleu, save_checkpoint, load_checkpoint
from torch.utils.tensorboard import SummaryWriter
from torchtext.legacy.datasets import Multi30k # German to English Dataset
from torchtext.legacy.data import Field, BucketIterator # For preprocessing
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


class Transformer(nn.Module):
    def __init__(
        self,
        embedding_size,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        forward_expansion,
        dropout,
        max_seq_length, # for position embedding
        device
    ):
        super(Transformer, self).__init__()
        # word token & position embedding for encoder
        self.src_word_embedding = nn.Embedding(src_vocab_size, embedding_size)
        self.src_position_embedding = nn.Embedding(max_seq_length, embedding_size)

        # word token & position embedding for decoder
        self.trg_word_embedding = nn.Embedding(trg_vocab_size, embedding_size)
        self.trg_position_embedding = nn.Embedding(max_seq_length, embedding_size)

        self.device = device
        self.transformer = nn.Transformer(
            d_model = embedding_size, # the number of expected features in the encoder/decoder inputs = embedding_size
            nhead = num_heads, # the number of heads in the multiheadattention
            num_encoder_layers = num_encoder_layers, # the number of sub-encoder-layers
            num_decoder_layers = num_decoder_layers, # the number of sub-decoder-layers
            dim_feedforward = forward_expansion, # the dimension of the feedforward network model (hidden -> dim_feed -> hidden)
            dropout = dropout, # the dropout value (default=0.1).
        )

        self.fc_out = nn.Linear(embedding_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.src_pad_idx = src_pad_idx

    # make source mask (no need to compute for padded values)
    def make_src_mask(self, src):
        # src shape: (src_len, N) -> for nn.Transformer, (N, src_len)
        src_mask = src.transpose(0, 1) == self.src_pad_idx
        return src_mask.to(self.device)

    def forward(self, src, trg):
        src_seq_length, N = src.shape
        trg_seq_length, N = trg.shape

        # make position value for position embeddings
        src_positions = (
            torch.arange(0, src_seq_length).unsqueeze(1).expand(src_seq_length, N).to(self.device)
        )
        trg_positions = (
            torch.arange(0, trg_seq_length).unsqueeze(1).expand(trg_seq_length, N).to(self.device)
        )

        # make embedded input token (word embedding + position embeding with dropout)
        embed_src = self.dropout(
            self.src_word_embedding(src) + self.src_position_embedding(src_positions)
        )
        embed_trg = self.dropout(
            self.trg_word_embedding(trg) + self.trg_position_embedding(trg_positions)
        )

        src_padding_mask = self.make_src_mask(src)
        trg_mask = self.transformer.generate_square_subsequent_mask(trg_seq_length).to(self.device) # triangle mask
        
        out = self.transformer(
            embed_src,
            embed_trg,
            src_key_padding_mask = src_padding_mask,
            tgt_mask = trg_mask
        )
        out = self.fc_out(out)

        return out

# Setup the training phase
load_model = False
save_model = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

# Training Hyperparameters
num_epochs = 5
learning_rate = 3e-4
batch_size = 32

# Model Hyperparameters
src_vocab_size = len(german.vocab)
trg_vocab_size = len(english.vocab)
embedding_size = 512
num_heads = 8
num_encoder_layers = 6
num_decoder_layers = 6
dropout = 0.1
max_seq_length = 100
forward_expansion = 512 * 4
src_pad_idx = english.vocab.stoi["<pad>"]

# Tensorboard for nice plots
writer = SummaryWriter('runs/loss_plot')
step = 0

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, validation_data, test_data),
    batch_size=batch_size,
    sort_within_batch=True,
    sort_key=lambda x: len(x.src), # prioritize to have examples that are similar lenght in batch -> want to minimize padding
    device=device)

model = Transformer(
    embedding_size,
    src_vocab_size,
    trg_vocab_size,
    src_pad_idx,
    num_heads,
    num_encoder_layers,
    num_decoder_layers,
    forward_expansion,
    dropout,
    max_seq_length,
    device
).to(device)


optimizer = optim.Adam(model.parameters(), lr=learning_rate)

pad_idx = english.vocab.stoi["<pad>"]
criterion = nn.CrossEntropyLoss(ignore_index = pad_idx)

if load_model:
    load_checkpoint(troch.load("my_checkpoint.pth.tar"), model, optimizer)

sentence = "ein pferd geht unter einer brücke neben einem boot."

for epoch in range(num_epochs):
    print(f"[Epoch {epoch} / {num_epochs}]")

    if save_model:
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        save_checkpoint(checkpoint)
    
    model.eval()
    translated_sentence = translate_sentence(
        model, sentence, german, english, device, max_length = 100
    )
    print(f"Translated example sentence \n {translated_sentence}")
    model.train()

    for batch_idx, batch in tqdm(enumerate(train_iterator)):
        # Get input and targets and get to cuda
        inp_data = batch.src.to(device)
        target = batch.trg.to(device)

        # forward propagation
        output = model(inp_data, target[:-1])
        # output shape: (trg_len, batch_size, trg_vocab_size)

        # Output is of shape (trg_len, batch_size, output_dim) but Cross Entropy Loss
        # doesn't take input in that form. For example if we have MNIST we want to have
        # output to be: (N, 10) and targets just (N). Here we can view it in a similar
        # way that we have output_words * batch_size that we want to send in into
        # our cost function, so we need to do some reshapin.
        # Let's also remove the start token while we're at it
        output = output.reshape(-1, output.shape[2]) # to make matrix
        target = target[1:].reshape(-1)

        optimizer.zero_grad()
        loss = criterion(output, target)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()

        writer.add_scalar('Training Loss', loss, global_step = step)
        step += 1


score = bleu(test_data[1:100], model, german, english, device)
print(f"Bleu score {score*100:.2f}")
    

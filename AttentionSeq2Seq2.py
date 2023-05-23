#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import csv
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
import math
import subprocess
subprocess.call(['pip', 'install', 'wandb'])
import wandb
wandb.login()
import argparse

# Check if CUDA is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[ ]:


import math
import time
import torch
import numpy as np

class Helper_Functions:
    @staticmethod
    def Time(s):
        # Converts seconds to minutes and seconds
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

    @staticmethod
    def Span(since, percent):
        # Calculates the elapsed time and estimated remaining time
        now = time.time()
        s = now - since
        es = s / percent
        rs = es - s
        return '%s (- %s)' % (Helper_Functions.Time(s), Helper_Functions.Time(rs))

    @staticmethod
    def key_printing(d, v):
        # Prints keys from a dictionary that have specific values
        f = [k for k, val in d.items() if val in v]
        if f:
            s = ''.join(f)
            if s[-1] == '\n':
                s = s[:-1]
            elif s[0] == '\t':
                s = s[1:]
        else:
            print("No keys")

        return s

    @staticmethod
    def char_acc(t, o):
        # Calculates character-level accuracy between predicted and target tensors
        with torch.no_grad():
            se = [[o[i][j].item() == t[i][j].item() for j in range(t.shape[1])] for i in range(t.shape[0])]
            c = np.sum(se)
            tc = np.sum([len(row) for row in se])
        return c / tc

    @staticmethod
    def word_acc(t, o):
        # Calculates word-level accuracy between predicted and target tensors
        o1 = torch.argmax(o, dim=1)
        with torch.no_grad():
            c = sum([(o1[i] == t[i]).sum().item() == t.shape[1] for i in range(t.shape[0])])
        return c / t.shape[0]

    @staticmethod
    def sample_equidi_pts(d, e):
        # Samples equidistant points from a given list
        s = len(d) // e
        i = np.arange(0, len(d), s)
        p = [d[j] for j in i]
        return p


HF = Helper_Functions()

# Creating a dictionary of helper functions
variables = {
    'asMinutes': HF.Time,
    'timeSince': HF.Span,
    'key_printing': HF.key_printing,
    'char_acc': HF.char_acc,
    'word_acc': HF.word_acc,
    'sample_equidi_pts': HF.sample_equidi_pts
}

# Accessing the variables
asMinutes = variables['asMinutes']
timeSince = variables['timeSince']
key_printing = variables['key_printing']
char_acc = variables['char_acc']
word_acc = variables['word_acc']
sample_equidi_pts = variables['sample_equidi_pts']

# Usage
# print(asMinutes)  # Prints the formatted time in minutes and seconds
# print(timeSince)  # Prints the elapsed time and estimated remaining time
# print(key_printing)  # Prints keys from a dictionary that have specific values
# print(char_acc)  # Calculates and returns character-level accuracy
# print(word_acc)  # Calculates and returns word-level accuracy
# print(sample_equidi_pts)  # Samples equidistant points from a given list


# In[ ]:


# Define a dictionary of variables and their corresponding characters
variables = {
    's_t_c_r': '\t',     # Represents the tab character
    'e_d_c_r': '\n',       # Represents the newline character
    'b_k_c_r': ' ',      # Represents a blank space character
    'u_n_c_r': '\r'    # Represents a carriage return character
}

# Assign the value of 's_t_c_r' from the 'variables' dictionary to the variable 's_t_c_r'
s_t_c_r = variables['s_t_c_r']

# Assign the value of 'e_d_c_r' from the 'variables' dictionary to the variable 'e_d_c_r'
e_d_c_r = variables['e_d_c_r']

# Assign the value of 'b_k_c_r' from the 'variables' dictionary to the variable 'b_k_c_r'
b_k_c_r = variables['b_k_c_r']

# Assign the value of 'u_n_c_r' from the 'variables' dictionary to the variable 'u_n_c_r'
u_n_c_r = variables['u_n_c_r']


# In[ ]:


import csv  # Import the csv module
import torch  # Import the torch module1

class DataPreProcessor:
    def __init__(self):
      self.file_paths = {
          'file1': '/home/user/Downloads/Swapnil/mar_test.csv',
          'file2': '/home/user/Downloads/Swapnil/mar_valid.csv',
          'file3': '/home/user/Downloads/Swapnil/mar_train.csv'
      }

      self.encodings = {
          'file1': 'utf-8',
          'file2': 'utf-8',
          'file3': 'utf-8'
      }

      self.variables = {
                    's_t_c_r': '\t',     # Represents the tab character
                    'e_d_c_r': '\n',       # Represents the newline character
                    'b_k_c_r': ' ',      # Represents a blank space character
                    'u_n_c_r': '\r'    # Represents a carriage return character
                }
                
    def read_csv_files(self):
      fr = {}  # Dictionary to store csv reader objects
      hd = {}  # Dictionary to store headers of each file
      dt = {}  # Dictionary to store data from each file

      for key, fp in self.file_paths.items():
          encoding = self.encodings.get(key)  # Get the encoding for the current file
          file = open(fp, encoding=encoding)  # Open the file with the given file path and encoding
          csv_reader = csv.reader(file)  # Create a csv reader object
          fr[key] = csv_reader  # Store the csv reader object in the dictionary
          hd[key] = next(csv_reader)  # Retrieve the headers from the csv reader
          dt[key] = []  # Initialize an empty list to store the data

          for row in csv_reader:
              dt[key].append(row)  # Append each row of data to the corresponding list

          file.close()  # Close the file after reading

      # Return the headers and data from each file
      return hd['file1'], hd['file2'], hd['file3'], dt['file1'], dt['file2'], dt['file3']


    def Reading_Data(self, lst):
        i = [pair[0] for pair in lst]  # Extract the first element from each pair in the given list
        t = [pair[1] for pair in lst]  # Extract the second element from each pair in the given list
        return i, t  # Return the extracted first elements as 'i' and second elements as 't'


    def Dict_lang(self, inputs, targets):
        i_dict = {}         # Dictionary to store character-to-index mapping for inputs
        max_i_length = 0    # Variable to track maximum input string length
        i_char = []         # List to store unique characters in inputs

        t_dict = {}         # Dictionary to store character-to-index mapping for targets
        max_t_length = 0    # Variable to track maximum target string length
        t_char = []         # List to store unique characters in targets

        # Encoding Inputs and updating i_dict
        for string in inputs:
            max_i_length = max(len(string), max_i_length)  # Update maximum input string length
            for char in string:
                if char not in i_dict:
                    i_dict[char] = len(i_char)   # Assign a unique index to each unique character
                    i_char.append(char)         # Store the unique character in the list

        if self.variables['b_k_c_r'] not in i_dict:
            i_dict[self.variables['b_k_c_r']] = len(i_char)   # Assign index to the blank character if not present
            i_char.append(self.variables['b_k_c_r'])

        i_dict[self.variables['u_n_c_r']] = len(i_char)     # Assign index to the unknown character
        i_char.append(self.variables['u_n_c_r'])

        if self.variables['s_t_c_r'] not in t_dict:
            t_dict[self.variables['s_t_c_r']] = len(t_char)   # Assign index to the start character if not present
            t_char.append(self.variables['s_t_c_r'])

        # Encoding Targets and updating t_dict
        for string in targets:
            max_t_length = max(len(string) + 2, max_t_length)    # Update maximum target string length
            for char in string:
                if char not in t_dict:
                    t_dict[char] = len(t_char)   # Assign a unique index to each unique character
                    t_char.append(char)         # Store the unique character in the list

        if self.variables['e_d_c_r'] not in t_dict:
            t_dict[self.variables['e_d_c_r']] = len(t_char)     # Assign index to the end character if not present
            t_char.append(self.variables['e_d_c_r'])

        if self.variables['b_k_c_r'] not in t_dict:
            t_dict[self.variables['b_k_c_r']] = len(t_char)   # Assign index to the blank character if not present
            t_char.append(self.variables['b_k_c_r'])

        return i_dict, max_i_length, i_char, t_dict, max_t_length, t_char

    def encoding_words(self, wl, ld, ml, l):
        encds = []   # List to store the encoded sequences
        for w in wl:
            encd = [ld[c] if c in ld else ld[self.variables['u_n_c_r']] for c in w]
            # Assign the index of each character in the word, or the index of unknown character if not present

            if l == 0:
                encd.extend([ld[self.variables['b_k_c_r']]] * (ml - len(encd)))
                # If 'l' is 0 (inputs), pad the sequence with blank character indices up to the maximum length
            if l == 1:
                encd = [ld[self.variables['s_t_c_r']]] + encd + [ld[self.variables['e_d_c_r']]]
                encd.extend([ld[self.variables['b_k_c_r']]] * (ml - len(encd)))
                # If 'l' is 1 (targets), add start and end character indices to the sequence and pad it with blank character indices up to the maximum length

            encds.append(encd)  # Append the encoded sequence to the list

        return encds  # Return the list of encoded sequences


    def Tokenzieations(self, train, val, test, input_dict, target_dict, max_input_length, max_target_length):
        t_i, t_t = self.Reading_Data(train)  # Reading train data
        te_i, te_t = self.Reading_Data(test)  # Reading test data
        v_i, v_t = self.Reading_Data(val)  # Reading validation data

        e_t_i = self.encoding_words(t_i, input_dict, max_input_length, 0)  # Encoding train inputs
        e_t_t = self.encoding_words(t_t, target_dict, max_target_length, 1)  # Encoding train targets
        e_v_i = self.encoding_words(v_i, input_dict, max_input_length, 0)  # Encoding validation inputs
        e_v_t = self.encoding_words(v_t, target_dict, max_target_length, 1)  # Encoding validation targets
        e_te_i = self.encoding_words(te_i, input_dict, max_input_length, 0)  # Encoding test inputs
        e_te_t = self.encoding_words(te_t, target_dict, max_target_length, 1)  # Encoding test targets

        return {
            'en_tr_ip': e_t_i,
            'en_tr_tr': e_t_t,
            'en_vl_ip': e_v_i,
            'en_vl_tr': e_v_t,
            'en_tt_ip': e_te_i,
            'en_tt_tr': e_te_t
        }

    def tensor_pair_conversion(self, tt_ip, tt_tr):
        pairs = [(torch.tensor(input_data), torch.tensor(target_data)) for input_data, target_data in
                zip(tt_ip, tt_tr)]  # Creating pairs of tensor inputs and targets
        return pairs

DPP = DataPreProcessor()

# Dictionary of functions
function_dict = {
    'tensor_pair_conversion': DPP.tensor_pair_conversion,
    'Tokenzieations': DPP.Tokenzieations,
    'encoding_words': DPP.encoding_words,
    'Reading_Data': DPP.Reading_Data,
    'Dict_lang': DPP.Dict_lang,
    'read_csv_files': DPP.read_csv_files
}

tensor_pair_conversion = function_dict['tensor_pair_conversion']
Tokenzieations = function_dict['Tokenzieations']
encoding_words = function_dict['encoding_words']
Reading_Data = function_dict['Reading_Data']
Dict_lang = function_dict['Dict_lang']
read_csv_files = function_dict['read_csv_files']

# Call the function and assign the returned values to variables
h1, h2, h3, test, val, train = read_csv_files()

tt_ip, tt_tr = Reading_Data(train)  # Reading train data
tst_ip, tst_tr = Reading_Data(test)  # Reading test data
vl_ip, vl_tr = Reading_Data(val)  # Reading validation data

# print(tt_ip[1])  # Print train input at index 1
# print(tt_tr[1])  # Print train target at index 1

input_dict, max_input_length, input_char, target_dict, max_target_length, target_char = Dict_lang(tt_ip + vl_ip + tst_ip, tt_tr + vl_tr + tst_tr)  # Generating dictionaries for inputs and targets

result = Tokenzieations(train, val, test, input_dict, target_dict, max_input_length, max_target_length)  # Tokenizing the data

en_tr_ip = result['en_tr_ip']  # Encoded train inputs
en_tr_tr = result['en_tr_tr']  # Encoded train targets
en_vl_ip = result['en_vl_ip']  # Encoded validation inputs
en_vl_tr = result['en_vl_tr']  # Encoded validation targets
en_tt_ip = result['en_tt_ip']  # Encoded test inputs
en_tt_tr = result['en_tt_tr']  # Encoded test targets

r = random.randint(0,100)  # Generate a random number between 0 and 100
# print(key_printing(input_dict, en_tr_ip[int(r)]))  # Print the keys corresponding to the values in input_dict for the randomly selected encoded train input
# print(key_printing(target_dict, en_tr_tr[int(r)]))  # Print the keys corresponding to the values in target_dict for the randomly selected encoded train target

en_tt_pa = tensor_pair_conversion(en_tr_ip, en_tr_tr)  # Convert encoded train inputs and targets into pairs of tensors
en_vl_pa = tensor_pair_conversion(en_vl_ip, en_vl_tr)  # Convert encoded validation inputs and targets into pairs of tensors
en_tst_pa = tensor_pair_conversion(en_tt_ip, en_tt_tr)  # Convert encoded test inputs and targets into pairs of tensors

pairs = (en_tt_pa, en_vl_pa, en_tst_pa)  # Combine pairs into a tuple
pair = random.choice(en_tt_pa)  # Randomly select a pair from the encoded train pairs

# print(key_printing(input_dict, pair[0]))  # Print the keys corresponding to the values in input_dict for the selected pair input
# print(key_printing(target_dict, pair[1]))  # Print the keys corresponding to the values in target_dict for the selected pair target


# ## Vanilla Seqence_2_Seqence_Network

# In[ ]:


class Recurrent_NN_Encoder(nn.Module):
    def __init__(self, device, cell_type, vocab_size, embed_dim, hidden_size, num_layers=1, bidirectional=False, dropout_p=0):
        super(Recurrent_NN_Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)  # Embedding layer for converting input indices to dense vectors
        self.hidden_size = hidden_size  # Size of the hidden state in the RNN
        self.num_layers = num_layers  # Number of layers in the RNN
        self.bidirectional = bidirectional  # Flag indicating whether the RNN is bidirectional or not
        self.cell_type = cell_type  # Type of RNN cell ('lstm', 'rnn', or 'gru')
        self.dropout_p = dropout_p  # Dropout probability
        self.dropout = nn.Dropout(self.dropout_p)  # Dropout layer for regularization
        
        # Initialize the RNN cell based on the specified cell type
        if cell_type == 'lstm':
            self.rnn = nn.LSTM(embed_dim, hidden_size, num_layers=self.num_layers, batch_first=True, dropout=dropout_p, bidirectional=self.bidirectional)
        elif cell_type == 'rnn':
            self.rnn = nn.RNN(embed_dim, hidden_size, num_layers=self.num_layers, batch_first=True, dropout=dropout_p, bidirectional=self.bidirectional)
        elif cell_type == 'gru':
            self.rnn = nn.GRU(embed_dim, hidden_size, num_layers=self.num_layers, batch_first=True, dropout=dropout_p, bidirectional=self.bidirectional)

    def forward(self, x, hidden, cell):
        out = self.embedding(x)  # Perform embedding lookup to get dense representations of input indices
        out = self.dropout(out)  # Apply dropout to the input embeddings
        if self.cell_type == 'lstm':
            out, (hidden, cell) = self.rnn(out, (hidden, cell))  # Forward pass through the LSTM
            return out, hidden, cell
        elif self.cell_type == 'rnn':
            out, hidden = self.rnn(out, hidden)  # Forward pass through the RNN
            return out, hidden
        elif self.cell_type == 'gru':
            out, hidden = self.rnn(out, hidden)  # Forward pass through the GRU
            return out, hidden
    
    def init_hidden(self, batch_size):
        # Initialize the hidden and cell states with random values
        hidden = torch.randn((1 + int(self.bidirectional)) * self.num_layers, batch_size, self.hidden_size, device=device)
        cell = torch.randn((1 + int(self.bidirectional)) * self.num_layers, batch_size, self.hidden_size, device=device)
        return hidden, cell


class Recurrent_NN_Decoder(nn.Module):
    def __init__(self, device, cell_type, output_vocab, embed_size, hidden_size, max_length, dropout_p=0.1, num_layers=1, bidirectional=False):
        super(Recurrent_NN_Decoder, self).__init__()
        self.hidden_size = hidden_size  # Hidden size of the decoder
        self.output_size = output_vocab  # Size of the output vocabulary
        self.embed_size = embed_size  # Size of the input embedding
        self.dropout_p = dropout_p  # Dropout probability
        self.cell_type = cell_type  # Type of RNN cell (lstm, gru, rnn)
        self.max_length = max_length  # Maximum length of input sequence
        self.device = device  # Device (e.g., CPU or GPU) to be used for computations
        self.num_layers = num_layers  # Number of layers in the decoder
        self.embedding_decoder = nn.Embedding(self.output_size, self.embed_size)  # Embedding layer for the decoder
        self.dropout = nn.Dropout(self.dropout_p)  # Dropout layer
        self.bidirectional = bidirectional  # Flag indicating if the encoder is bidirectional

        # Determine the type of RNN cell based on the given cell_type
        if cell_type == 'lstm':
            self.rnn = nn.LSTM(self.embed_size, hidden_size, num_layers=self.num_layers, batch_first=True, bidirectional=self.bidirectional, dropout=self.dropout_p)
        elif cell_type == 'gru':
            self.rnn = nn.GRU(self.embed_size, hidden_size, num_layers=self.num_layers, batch_first=True, bidirectional=self.bidirectional, dropout=self.dropout_p)
        elif cell_type == 'rnn':
            self.rnn = nn.RNN(self.embed_size, hidden_size, num_layers=self.num_layers, batch_first=True, bidirectional=self.bidirectional, dropout=self.dropout_p)

        self.out = nn.Linear((1 + int(self.bidirectional)) * self.hidden_size, self.output_size)  # Linear layer for output prediction
        self.out_activation = nn.LogSoftmax(dim=-1)  # Log softmax activation for output probabilities

    def forward(self, input, hidden, cell):
        input = input.unsqueeze(1)  # Add a singleton dimension to the input tensor
        embedded_decoder = self.embedding_decoder(input)  # Apply embedding to the input
        embedded_decoder = self.dropout(embedded_decoder)  # Apply dropout to the embedded input

        # Pass the embedded input through the RNN cell
        if self.cell_type == 'lstm':
            output, (hidden, cell) = self.rnn(embedded_decoder, (hidden, cell))
        elif self.cell_type == 'gru':
            output, hidden = self.rnn(embedded_decoder, hidden)
        elif self.cell_type == 'rnn':
            output, hidden = self.rnn(embedded_decoder, hidden)

        output = F.relu(self.out(output))  # Apply ReLU activation to the output
        output = F.log_softmax(output, dim=-1)  # Apply log softmax activation to obtain output probabilities

        return output, hidden, cell  # Return the output, hidden state, and cell state

    def init_hidden(self, encoder_hidden, encoder_cell, encoder_bidirectional):
        hidden = encoder_hidden[-(1 + int(encoder_bidirectional)): ].repeat(self.num_layers, 1, 1)  # Initialize the hidden state using the encoder's hidden state
        cell = encoder_cell[-(1 + int(encoder_bidirectional)): ].repeat(self.num_layers, 1, 1)  # Initialize the cell state using the encoder's cell state
        return hidden, cell  # Return the initialized hidden state and cell state



class Seqence_2_Seqence_Network(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder  # Initialize the encoder
        self.decoder = decoder  # Initialize the decoder
        self.device = device  # Store the device (e.g., CPU or GPU)
        self.max_target_length = 0  # Variable to store the maximum target sequence length
        self.sos = 0  # Start-of-sequence token
        
    def forward(self, source, target, teacher_forcing_ratio=0.5):
        batch_size = target.shape[0]  # Get the batch size
        target_len = target.shape[1]  # Get the target sequence length
        self.max_target_length = target_len  # Update the maximum target sequence length
        target_vocab_size = self.decoder.output_size  # Get the target vocabulary size
        
        outputs = torch.zeros(batch_size, target_len, target_vocab_size).to(self.device)  # Initialize the outputs tensor
        
        encoder_hidden, encoder_cell = self.encoder.init_hidden(batch_size)  # Initialize the encoder hidden state and cell state
        
        if (self.encoder.cell_type == 'lstm'):
            encoder_outputs, encoder_hidden, encoder_cell = self.encoder.forward(source, encoder_hidden, encoder_cell)
            # Forward pass through the encoder LSTM
        if (self.encoder.cell_type == 'rnn'):
            encoder_outputs, encoder_hidden = self.encoder.forward(source, encoder_hidden, encoder_cell)
            # Forward pass through the encoder RNN
        if (self.encoder.cell_type == 'gru'):
            encoder_outputs, encoder_hidden = self.encoder.forward(source, encoder_hidden, encoder_cell)
            # Forward pass through the encoder GRU
        
        input = target[:, 0]  # Set the first input to the decoder as the <sos> token
        self.sos = target[:, 0]  # Store the <sos> token
        
        hidden, cell = self.decoder.init_hidden(encoder_hidden, encoder_cell, self.encoder.bidirectional)
        # Initialize the decoder hidden state and cell state
        
        for t in range(1, target_len):
            output, hidden, cell = self.decoder.forward(input, hidden, cell)
            # Forward pass through the decoder
            outputs[:, t] = output.squeeze(1)  # Store the decoder output in the outputs tensor
            teacher_force = random.random() < teacher_forcing_ratio  # Determine whether to use teacher forcing
            top1 = output.argmax(-1)  # Get the index of the highest probability output
            input = target[:, t] if teacher_force else top1.squeeze(1)  # Set the next input to the decoder
            
        return outputs
    
    def inference(self, source, target):
        batch_size = source.shape[0]  # Get the batch size
        target_len = self.max_target_length  # Get the maximum target sequence length
        target_vocab_size = self.decoder.output_size  # Get the target vocabulary size
        
        outputs = torch.zeros(batch_size, target_len, target_vocab_size).to(self.device)  # Initialize the outputs tensor
        
        encoder_hidden, encoder_cell = self.encoder.init_hidden(batch_size)  # Initialize the encoder hidden state and cell state
        
        if (self.encoder.cell_type == 'lstm'):
            encoder_outputs, encoder_hidden, encoder_cell = self.encoder.forward(source, encoder_hidden, encoder_cell)
            # Forward pass through the encoder LSTM
        if (self.encoder.cell_type == 'rnn'):
            encoder_outputs, encoder_hidden = self.encoder.forward(source, encoder_hidden, encoder_cell)
            # Forward pass through the encoder RNN
        if (self.encoder.cell_type == 'gru'):
            encoder_outputs, encoder_hidden = self.encoder.forward(source, encoder_hidden, encoder_cell)
            # Forward pass through the encoder GRU
        
        input = self.sos  # Set the first input to the <sos> token
        
        hidden, cell = self.decoder.init_hidden(encoder_hidden, encoder_cell, self.encoder.bidirectional)
        # Initialize the decoder hidden state and cell state
        
        for t in range(1, target_len):
            output, hidden, cell = self.decoder.forward(input, hidden, cell)
            # Forward pass through the decoder
            outputs[:, t] = output.squeeze(1)  # Store the decoder output in the outputs tensor
            top1 = output.argmax(-1)  # Get the index of the highest probability output
            input = top1.squeeze(1)  # Set the next input to the decoder
            
        return outputs





# ## Vanilla Model

# In[ ]:


def vanilla_model(input_dict, target_dict):
    variables = {}
    
    # Count the number of elements in the input and target dictionaries
    variables['a'] = len(input_dict)
    variables['b'] = len(target_dict)
    
    # Set values for variables c, d, e, f, g, h, i, j, k, and max_length
    variables['c'] = 32    #batch size
    variables['d'] = 32     #val batch size
    variables['e'] = 256    #enc_embedding
    variables['f'] = 256    #dec embedding
    variables['g'] = 512    #hidden_size 
    variables['h'] = 3     #enc_number_layers
    variables['i'] = 2     #dec_num_layers
    variables['j'] = 0.4    #enc_dropout
    variables['k'] = 0.4    #dec_drop_out
    variables['max_length'] = max_target_length
    
    # Set value for variable l
    variables['l'] = 'gru'
    
    # Create an instance of Recurrent_NN_Encoder and assign it to variable m
    variables['m'] = Recurrent_NN_Encoder(device, variables['l'], variables['a'], variables['e'], variables['g'],
                                variables['h'], bidirectional=True, dropout_p=variables['j'])
    
    # Create an instance of Recurrent_NN_Decoder and assign it to variable n
    variables['n'] = Recurrent_NN_Decoder(device, variables['l'], variables['b'], variables['f'], variables['g'],
                                variables['max_length'], variables['k'], variables['i'], bidirectional=True)
    
    # Create an instance of Seqence_2_Seqence_Network by passing in the encoder and decoder instances, and assign it to variable o
    variables['o'] = Seqence_2_Seqence_Network(variables['m'], variables['n'], device).to(device)
    
    # Define a function count_parameters(model) that returns the total number of trainable parameters in a model
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Print the number of trainable parameters in the Seqence_2_Seqence_Network model
    print(f' {count_parameters(variables["o"]):,} trainable parameters')
    
    # Create an instance of Adam optimizer and assign it to variable p
    variables['p'] = torch.optim.Adam(variables['o'].parameters(), lr=0.001)
    
    # Create an instance of negative log likelihood loss and assign it to variable q
    variables['q'] = nn.NLLLoss()
    
    # Return the Seqence_2_Seqence_Network model, optimizer, and criterion
    return variables['o'], variables['p'], variables['q']

# Call the vanilla_model function with input_dict and target_dict, and assign the returned values to model, optimizer, and criterion respectively
model, optimizer, criterion = vanilla_model(input_dict, target_dict)


# In[ ]:


# def train_function(model, pairs, batch_size, n_iters, optimizer, tf, print_every=10, plot_every=10, log = True, Attention = False):

def train_function(m, p, b, n, o, t, p_e=10, p_v=10, l=True, A=False):
    s = time.time()  # Starting time

    p_l = []  # List to store training loss
    t_w_a = []  # List to store word-level accuracy on training set
    v_w_a = []  # List to store word-level accuracy on validation set
    p_l_t = 0  # Training loss per iteration
    p_l_v_t = 0  # Validation loss per iteration

    t_p = p[0]  # Training data
    v_p = p[1]  # Validation data
    t_a = 0  # Accumulated word-level accuracy on training set

    c = nn.NLLLoss()  # Negative log likelihood loss
    wandb.init()
    count = 0  # Iteration count
    for it in range(1, n + 1):  # Number of iterations
        for i in range(0, len(t_p) - b, b):  # Batch processing
            t_a = 0  # Reset accumulated word-level accuracy
            count += 1  # Increment iteration count

            if i + b > len(t_p):
                b = len(t_p) - i + 1  # Adjust batch size if remaining samples are less than the batch size

            i_t = torch.stack([t_p[i + j][0] for j in range(b)]).squeeze(1).long().cuda()  # Input tensor
            t_t = torch.stack([t_p[i + j][1] for j in range(b)]).squeeze(1).long().cuda()  # Target tensor

            o.zero_grad()  # Clear gradients
            out = m(i_t, t_t, teacher_forcing_ratio=t if count < 4000 else 0)  # Forward pass
            out = torch.permute(out, [0, 2, 1])  # Permute output tensor dimensions
            l = c(out, t_t)  # Calculate loss

            t_a_w = word_acc(t_t, out) * b  # Calculate word-level accuracy on training set
            t_a += t_a_w  # Accumulate word-level accuracy
            l.backward()  # Backward pass
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1)  # Clip gradients to avoid exploding gradients
            o.step()  # Update model parameters

            p_l_t += l  # Accumulate training loss per iteration
            p_l_v_t += l  # Accumulate validation loss per iteration

            if count % 800 == 0:
                v_i_t = torch.stack([v_p[j][0] for j in range(b)]).squeeze(1).long().cuda()  # Validation input tensor
                v_t_t = torch.stack([v_p[j][1] for j in range(b)]).squeeze(1).long().cuda()  # Validation target tensor

                if A:
                    v_o, _ = m.inference(v_i_t, v_t_t)
                else:
                    v_o = m.inference(v_i_t, v_t_t)

                v_o = v_o.permute(0, 2, 1)  # Permute validation output tensor dimensions
                v_l = c(v_o, v_t_t)  # Calculate validation loss
                v_l_s = v_l
                wandb.log({'Val_Loss': v_l_s})  # Log validation loss

            if count % 800 == 0:
                p_l_a = p_l_t / 800  # Average training loss over 800 iterations
                p_l_t = 0  # Reset training loss accumulator
                print('%s (%d %d%%) %.7f' % (timeSince(s, it / n), it, it / n * 100, p_l_a))  # Print progress

            if count % 800 == 0:
                p_l_a = p_l_v_t / 800  # Average validation loss over 800 iterations
                p_l.append(p_l_a.detach())  # Append training loss to list
                wandb.log({'Train Loss': p_l_a})  # Log training loss
                p_l_v_t = 0  # Reset validation loss accumulator

        t_a = t_a / (len(t_p) - b)  # Calculate average word-level accuracy on training set
        t_w_a.append(t_a)  # Append word-level accuracy to list

    print(t_w_a)  # Print word-level accuracy on training set

    p_l = [l.cpu().numpy() for l in p_l]  # Convert training loss to numpy array
    p_l_s = sample_equidi_pts(p_l, n)  # Sample equidistant points from the training loss

    w_c = 0  # Total correct predictions
    for i in range(0, len(v_p) - b, b):  # Batch processing on validation set
        if i + b > len(v_p):
            b = len(v_p) - i + 1  # Adjust batch size if remaining samples are less than the batch size

        v_i_t = torch.stack([v_p[i + j][0] for j in range(b)]).squeeze(1).long().cuda()  # Validation input tensor
        v_t_t = torch.stack([v_p[i + j][1] for j in range(b)]).squeeze(1).long().cuda()  # Validation target tensor

        if A:
            v_o, _ = m.inference(v_i_t, v_t_t)
        else:
            v_o = m.inference(v_i_t, v_t_t)

        v_o = v_o.permute(0, 2, 1)  # Permute validation output tensor dimensions
        v_l = c(v_o, v_t_t)  # Calculate validation loss

        v_a_w = word_acc(v_t_t, v_o)  # Calculate word-level accuracy on validation set
        w_c += v_a_w * b  # Accumulate correct predictions

    w_a = w_c / (len(v_p) - b)  # Average word-level accuracy on validation set
    m = {'Val_Accuracy': w_a}  # Log validation accuracy
    wandb.log(m)

    print(f"Val loss = {v_l}")  # Print validation loss
    print(f'Word-level-accuracy on val set = {w_a}')  # Print word-level accuracy on validation set


# In[ ]:


class AttentionRecurrent_NN_Decoder(nn.Module):
    def __init__(self, device, cell_type, output_vocab, embed_size, hidden_size, max_length, dropout_p=0.1, num_layers=1, bidirectional=False):
        super(AttentionRecurrent_NN_Decoder, self).__init__()
        self.hidden_size = hidden_size  # Hidden size of the decoder
        self.output_size = output_vocab  # Size of the output vocabulary
        self.embed_size = embed_size  # Size of the input embedding
        self.dropout_p = dropout_p  # Dropout probability
        self.cell_type = cell_type  # Type of RNN cell (lstm, gru, rnn)
        self.max_length = max_length  # Maximum length of input sequence
        self.device = device  # Device (e.g., CPU or GPU) to be used for computations
        self.num_layers = num_layers  # Number of layers in the decoder
        self.embedding_decoder = nn.Embedding(self.output_size, self.embed_size)  # Embedding layer for the decoder
        self.dropout = nn.Dropout(self.dropout_p)  # Dropout layer
        self.bidirectional = bidirectional  # Flag indicating if the encoder is bidirectional

        # Determine the type of RNN cell based on the given cell_type
        if cell_type == 'lstm':
            self.rnn = nn.LSTM(self.embed_size + hidden_size * (1 + int(self.bidirectional)), hidden_size, num_layers=self.num_layers, batch_first=True, bidirectional=self.bidirectional, dropout=self.dropout_p)
        elif cell_type == 'gru':
            self.rnn = nn.GRU(self.embed_size + hidden_size * (1 + int(self.bidirectional)), hidden_size, num_layers=self.num_layers, batch_first=True, bidirectional=self.bidirectional, dropout=self.dropout_p)
        elif cell_type == 'rnn':
            self.rnn = nn.RNN(self.embed_size + hidden_size * (1 + int(self.bidirectional)), hidden_size, num_layers=self.num_layers, batch_first=True, bidirectional=self.bidirectional)

        self.energy = nn.Linear(hidden_size * (2 + int(self.bidirectional)), hidden_size)  # Linear layer for energy calculation
        self.value = nn.Linear(hidden_size, 1, bias=False)  # Linear layer for value calculation
        self.softmax = nn.Softmax(dim=0)  # Softmax activation for attention weights
        self.relu = nn.ReLU()  # ReLU activation function
        self.tanh = nn.Tanh()  # Hyperbolic tangent activation function
        
        self.out = nn.Linear((1 + int(self.bidirectional)) * self.hidden_size, self.output_size)  # Linear layer for output prediction
        self.out_activation = nn.LogSoftmax(dim=-1)  # Log softmax activation for output probabilities
  
        self.hidden_reshape_linear = nn.Linear(hidden_size * 2, hidden_size)  # Linear layer for reshaping the hidden state

    def forward(self, input, encoder_states, hidden, cell):
        input = input.unsqueeze(1)  # Add a singleton dimension to the input tensor
        embedded_decoder = self.embedding_decoder(input)  # Apply embedding to the input
        embedded_decoder = self.dropout(embedded_decoder)  # Apply dropout to the embedded input

        encoder_states = encoder_states.permute(1, 0, 2)  # Permute the encoder states tensor
        sequence_length = encoder_states.shape[0]  # Obtain the sequence length
        if self.bidirectional:
            hidden_1 = self.relu(self.hidden_reshape_linear(hidden[0:2].permute(1, 0, 2).reshape(hidden.shape[1], -1))).unsqueeze(0)
        else:
            hidden_1 = hidden[0]

        hidden_reshaped = hidden_1.repeat(sequence_length, 1, 1)  # Reshape and repeat the hidden state

        energy = self.value(self.tanh(self.energy(torch.cat((hidden_reshaped, encoder_states), dim=2))))  # Compute the energy
        attention = self.softmax(energy)  # Apply softmax to obtain attention weights
        attention = attention.permute(1, 2, 0)  # Permute the attention tensor
        encoder_states = encoder_states.permute(1, 0, 2)  # Permute the encoder states tensor
        context_vector = torch.bmm(attention, encoder_states)  # Compute the context vector

        rnn_input = torch.cat((context_vector, embedded_decoder), dim=2)  # Concatenate the context vector and embedded decoder input
      
        if self.cell_type == 'lstm':
            decoder_output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))  # Pass the input through the RNN cell
        elif self.cell_type == 'gru':
            decoder_output, hidden = self.rnn(rnn_input, hidden)  # Pass the input through the RNN cell
        elif self.cell_type == 'rnn':
            decoder_output, hidden = self.rnn(rnn_input, hidden)  # Pass the input through the RNN cell

        output = F.relu(self.out(decoder_output))  # Apply ReLU activation to the output
        output = F.log_softmax(output, dim=-1)  # Apply log softmax activation to obtain output probabilities

        return output, hidden, cell, attention  # Return the output, hidden state, cell state, and attention weights

    def init_hidden(self, encoder_hidden, encoder_cell, encoder_bidirectional):
        hidden = encoder_hidden[-(1 + int(encoder_bidirectional)): ].repeat(self.num_layers, 1, 1)  # Initialize the hidden state using the encoder's hidden state
        cell = encoder_cell[-(1 + int(encoder_bidirectional)): ].repeat(self.num_layers, 1, 1)  # Initialize the cell state using the encoder's cell state
        return hidden, cell  # Return the initialized hidden state and cell state


# In[ ]:


class AttentionSeqence_2_Seqence_Network(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder  # Initialize the encoder
        self.decoder = decoder  # Initialize the decoder
        self.device = device  # Store the device (e.g., CPU or GPU)
        self.max_target_length = 0  # Variable to store the maximum target sequence length
        self.sos = 0  # Start-of-sequence token
               
    def forward(self, source, target, teacher_forcing_ratio=0.5):
        batch_size = target.shape[0]  # Get the batch size
        target_len = target.shape[1]  # Get the target sequence length
        self.max_target_length = target_len  # Update the maximum target sequence length
        target_vocab_size = self.decoder.output_size  # Get the target vocabulary size
        
        outputs = torch.zeros(batch_size, target_len, target_vocab_size).to(self.device)  # Initialize the outputs tensor

        encoder_hidden, encoder_cell = self.encoder.init_hidden(batch_size)  # Initialize the encoder hidden state and cell state

        if (self.encoder.cell_type == 'lstm'):
            encoder_outputs, encoder_hidden, encoder_cell = self.encoder.forward(source, encoder_hidden, encoder_cell)
            # Forward pass through the encoder LSTM
        if (self.encoder.cell_type == 'rnn'):
            encoder_outputs, encoder_hidden = self.encoder.forward(source, encoder_hidden, encoder_cell)
            # Forward pass through the encoder RNN
        if (self.encoder.cell_type == 'gru'):
            encoder_outputs, encoder_hidden = self.encoder.forward(source, encoder_hidden, encoder_cell)
            # Forward pass through the encoder GRU

        input = target[:, 0]  # Set the first input to the decoder as the <sos> token
        self.sos = target[:, 0]  # Store the <sos> token
        hidden, cell = self.decoder.init_hidden(encoder_hidden, encoder_cell, self.encoder.bidirectional)
        # Initialize the decoder hidden state and cell state
        
        for t in range(1, target_len):
            output, hidden, cell, _ = self.decoder.forward(input, encoder_outputs, hidden, cell)
            # Forward pass through the decoder with attention
            outputs[:, t] = output.squeeze(1)  # Store the decoder output in the outputs tensor
            teacher_force = random.random() < teacher_forcing_ratio  # Determine whether to use teacher forcing
            top1 = output.argmax(-1)  # Get the index of the highest probability output
            input = target[:, t] if teacher_force else top1.squeeze(1)  # Set the next input to the decoder
            
        return outputs

    def inference(self, source, target):
        batch_size = source.shape[0]  # Get the batch size
        target_len = self.max_target_length  # Get the maximum target sequence length
        target_vocab_size = self.decoder.output_size  # Get the target vocabulary size
        
        outputs = torch.zeros(batch_size, target_len, target_vocab_size).to(self.device)  # Initialize the outputs tensor

        encoder_hidden, encoder_cell = self.encoder.init_hidden(batch_size)  # Initialize the encoder hidden state and cell state

        if (self.encoder.cell_type == 'lstm'):
            encoder_outputs, encoder_hidden, encoder_cell = self.encoder.forward(source, encoder_hidden, encoder_cell)
            # Forward pass through the encoder LSTM
        if (self.encoder.cell_type == 'rnn'):
            encoder_outputs, encoder_hidden = self.encoder.forward(source, encoder_hidden, encoder_cell)
            # Forward pass through the encoder RNN
        if (self.encoder.cell_type == 'gru'):
            encoder_outputs, encoder_hidden = self.encoder.forward(source, encoder_hidden, encoder_cell)
            # Forward pass through the encoder GRU

        input = self.sos  # Set the first input to the <sos> token
        input_len = encoder_outputs.shape[1]  # Get the length of the encoder outputs
        hidden, cell = self.decoder.init_hidden(encoder_hidden, encoder_cell, self.encoder.bidirectional)
        attention_map = torch.zeros(batch_size, target_len, input_len)  # Initialize the attention map
        
        for t in range(1, target_len):
            output, hidden, cell, attention = self.decoder.forward(input, encoder_outputs, hidden, cell)
            # Forward pass through the decoder with attention
            attention_map[:, t - 1, :] = attention.squeeze(1)  # Store the attention weights in the attention map
            outputs[:, t] = output.squeeze(1)  # Store the decoder output in the outputs tensor
            top1 = output.argmax(-1)  # Get the index of the highest probability output
            input = top1.squeeze(1)  # Set the next input to the decoder
            
        return outputs, attention_map


# In[ ]:


import argparse
import torch
import torch.nn as nn
import wandb
wandb.init()
def train_model(args):
    # Creating the validation data dictionary
    vd = {
        'input_dim': len(input_dict),
        'output_dim': len(target_dict),
        'batch_size': 32,
        'val_batch_size': 32,
        'enc_embedding': args.enc_embedding,
        'dec_embedding': args.dec_embedding,
        'hidden': args.hidden_size,
        'enc_num_layers': args.enc_layers,
        'dec_num_layers': args.dec_layers,
        'enc_dropout': args.dropout,
        'dec_dropout': args.dropout,
        'max_length': max_target_length,
        'cell_type': args.cell_type,
        'attention': args.Attention
    }

    # Extracting values from the validation data dictionary
    input_dim = vd['input_dim']
    output_dim = vd['output_dim']
    batch_size = vd['batch_size']
    val_batch_size = vd['val_batch_size']
    enc_embedding = vd['enc_embedding']
    dec_embedding = vd['dec_embedding']
    hidden = vd['hidden']
    enc_num_layers = vd['enc_num_layers']
    dec_num_layers = vd['dec_num_layers']
    enc_dropout = vd['enc_dropout']
    dec_dropout = vd['dec_dropout']
    max_length = vd['max_length']
    cell_type = vd['cell_type']
    attention = vd['attention']

    # Creating the encoder and decoder models
    enc = Recurrent_NN_Encoder(device, cell_type, input_dim, enc_embedding, hidden, enc_num_layers,
                     bidirectional=args.bidirectional, dropout_p=enc_dropout)
    if attention:
        dec = AttentionRecurrent_NN_Decoder(device, cell_type, output_dim, dec_embedding, hidden, max_length,
                                            dec_dropout, dec_num_layers, bidirectional=args.bidirectional)
        model = AttentionSeqence_2_Seqence_Network(enc, dec, device).to(device)
    else:
        dec = Recurrent_NN_Decoder(device, cell_type, output_dim, dec_embedding, hidden, max_length, dec_dropout,
                                    dec_num_layers, bidirectional=args.bidirectional)
        model = Seqence_2_Seqence_Network(enc, dec, device).to(device)

    # Setting the experiment name based on the config
    m = '_bi_' if args.bidirectional else '_uni_'
    a = '_Attention' if attention else '_'



    # Assigning the experiment name to the wandb run



    # Creating the optimizer based on the chosen optimiser
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) if args.optimiser == 'adam' else torch.optim.NAdam(
        model.parameters(), lr=0.001)

    criterion = nn.NLLLoss()

    # Training the model using the train_function function
    train_function(model, pairs, 32, args.epochs, optimizer, args.teacher_forcing_ratio, A=attention)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimiser", choices=['adam', 'nadam'], default='adam')
    parser.add_argument("--teacher_forcing_ratio", type=float, choices=[0.3, 0.5, 0.7], default=0.7)
    parser.add_argument("--bidirectional", type=bool, choices=[True, False], default=True)
    parser.add_argument("--enc_embedding", type=int, choices=[128, 256], default=128)
    parser.add_argument("--dec_embedding", type=int, choices=[128, 256], default=128)
    parser.add_argument("--hidden_size", type=int, choices=[64, 128, 256, 512], default=64)
    parser.add_argument("--enc_layers", type=int, choices=[2, 3], default=2)
    parser.add_argument("--dec_layers", type=int, choices=[2, 3], default=2)
    parser.add_argument("--dropout", type=float, choices=[0.25, 0.3, 0.4], default=0.25)
    parser.add_argument("--cell_type", choices=['gru', 'lstm', 'rnn'], default='gru')
    parser.add_argument("--Attention", type=bool, default=True)
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()

    train_model(args)



# Default=
# python AttentionSeq2Seq.py

# Other=
# python script.py --optimiser nadam --teacher_forcing_ratio 0.7 --bidirectional False --enc_embedding 256 --dec_embedding 256 --epochs 20 --hidden_size 512 --enc_layers 3 --dec_layers 3 --dropout 0.4 --cell_type gru --Attention True
# python script.py --optimiser adam --dropout 0.3


# In[ ]:


# # Function to train the model
# def train_model(args):
#     # Creating the validation data dictionary
#     vd = {
#         'input_dim': len(input_dict),
#         'output_dim': len(target_dict),
#         'batch_size': 32,
#         'val_batch_size': 32,
#         'enc_embedding': args.enc_embedding,
#         'dec_embedding': args.dec_embedding,
#         'hidden': args.hidden_size,
#         'enc_num_layers': args.enc_layers,
#         'dec_num_layers': args.dec_layers,
#         'enc_dropout': args.dropout,
#         'dec_dropout': args.dropout,
#         'max_length': max_target_length,
#         'cell_type': args.cell_type
#     }

#     # Extracting values from the validation data dictionary
#     input_dim = vd['input_dim']
#     output_dim = vd['output_dim']
#     batch_size = vd['batch_size']
#     val_batch_size = vd['val_batch_size']
#     enc_embedding = vd['enc_embedding']
#     dec_embedding = vd['dec_embedding']
#     hidden = vd['hidden']
#     enc_num_layers = vd['enc_num_layers']
#     dec_num_layers = vd['dec_num_layers']
#     enc_dropout = vd['enc_dropout']
#     dec_dropout = vd['dec_dropout']
#     max_length = vd['max_length']
#     cell_type = vd['cell_type']

#     # Creating the encoder and decoder models
#     enc = Recurrent_NN_Encoder(device, cell_type, input_dim, enc_embedding, hidden, enc_num_layers,
#                      bidirectional=args.bidirectional, dropout_p=enc_dropout)
#     dec = Recurrent_NN_Decoder(device, cell_type, output_dim, dec_embedding, hidden, max_length, dec_dropout,
#                      dec_num_layers, bidirectional=args.bidirectional)
#     model = Seqence_2_Seqence_Network(enc, dec, device).to(device)

#     # Setting the experiment name based on the config
#     exp_name = f"{args.cell_type}_e_{args.optimiser}"

#     # Assigning the experiment name to the wandb run
#     wandb.run.name = exp_name

#     # Creating the optimizer based on the chosen optimiser
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001) if args.optimiser == 'adam' else torch.optim.NAdam(
#         model.parameters(), lr=0.001)

#     criterion = nn.NLLLoss()

#     # Training the model using the train_function function
#     train_function(model, pairs, 32, args.epochs, optimizer, args.teacher_forcing_ratio)

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--optimiser", choices=['adam', 'nadam'], required=True)
#     parser.add_argument("--teacher_forcing_ratio", type=float, choices=[0.3, 0.5, 0.7], required=True)
#     parser.add_argument("--bidirectional", type=bool, choices=[True, False], required=True)
#     parser.add_argument("--enc_embedding", type=int, choices=[128, 256], required=True)
#     parser.add_argument("--dec_embedding", type=int, choices=[128, 256], required=True)
#     parser.add_argument("--epochs", type=int, choices=[1], required=True)
#     parser.add_argument("--hidden_size", type=int, choices=[64, 128, 256, 512], required=True)
#     parser.add_argument("--enc_layers", type=int, choices=[2, 3], required=True)
#     parser.add_argument("--dec_layers", type=int, choices=[2, 3], required=True)
#     parser.add_argument("--dropout", type=float, choices=[0.3, 0.4], required=True)
#     parser.add_argument("--cell_type", choices=['lstm', 'gru', 'rnn'], required=True)

#     args = parser.parse_args()

#     # Initializing wandb run
#     wandb.init()

#     # Training the model
#     train_model(args)

#     # Finishing the wandb run
#     wandb.finish()


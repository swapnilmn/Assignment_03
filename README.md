# Assignment_03

The objective of the assignment is as follows:

(i) Acquire knowledge in employing Recurrent Neural Networks to model sequence-to-sequence learning problems.

(ii) Analyze and contrast various cell types, including vanilla RNN, LSTM, and GRU.

(iii) Gain insights into how attention networks surpass the constraints of the basic seq2seq model.


#Problem Statement:

In this assignment, I work with the Aksharantar dataset from AI4Bharat. The dataset consists of pairs, each including a word in the native script and its transliteration in the Latin hand. My task was to train a model that can map a Romanized string to its corresponding word in Devanagari script, essentially performing character-level translation between languages.

# Data Pre Processing: 

The provided code performs various data processing operations. Here's a summary of what each section does:

1. Data_Reading function: Reads data from a list of pairs and separates the first and second elements into separate lists.

2. Reading data: Uses the Data_Reading function to read data from the 'train,' 'test,' and 'val' lists into separate input and target lists.

3. Defining a dictionary: Defines a dictionary that stores variable values, such as special characters represented by certain keys.

4. dictlang function: Creates dictionaries for language encoding based on the input and target data. It generates dictionaries, calculates the maximum length of input and target languages, and creates lists of characters in each language.

5. encoding_w function: Encodes words based on a given dictionary. It converts characters in each word to their corresponding dictionary values and handles padding and special characters.

6. tokenize function: Tokenizes and encodes the training, validation, and testing data using the dictionaries and encoding_w function.
Each section contains additional code that is commented out, which may be used for printing or further analysis.


# Question 1:
 1. EncoderRNN class:
Embeds input sequences using an embedding layer.
Passes the embedded input through an RNN (LSTM, GRU, or RNN).
Returns the RNN output, final hidden state, and final cell state.

2. DecoderRNN class:
Embeds input sequences using an embedding layer.
Passes the embedded input through an RNN (LSTM, GRU, or RNN).
Applies linear transformation and log softmax activation to generate output probabilities.

3. Seq2Seq class:
Combines the encoder and decoder components into a seq2seq model.
Performs encoding of source sequences using the encoder.
Generates target sequences using the decoder with teacher forcing during training.
Performs inference by feeding the predicted output from the previous time step as input for the next time step.

# Question 2: 

Here I have taken following hyperparameter choices
![image](https://github.com/swapnilmn/Assignment_03/assets/126043206/14c13663-b895-4a24-a15d-99f40d5bd635)



































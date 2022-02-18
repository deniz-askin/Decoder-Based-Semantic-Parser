# Decoder-Based Semantic Parser

A Decoder-only Transformer that can be tested on 4 different benchmark semantic parsing datasets (ATIS, GeoQuery, DJANGO, Jobs640). 

The original Code is from https://colab.research.google.com/github/dlmacedo/starter-academic/blob/master/content/courses/deeplearning/notebooks/tensorflow/transformer.ipynb

Modifications to the code include:
    
    Dropping the Encoder to make it a Decoder-only mechanism.
    
    Added options to use recurrent weights (GRU and LSTM) for the attention weights 
    and/or the hidden layer weights as well as an option to use label smoothing.
    
    An evaluator for the dataset that creates .txt files containing the accuracy of the test, 
    the shuffled train and test sets, and a list of the correct and wrong parses the engine produced.

Simply change the pathfiles to the dataset you would like to use in the file and set the hyperparameters you would like to use.

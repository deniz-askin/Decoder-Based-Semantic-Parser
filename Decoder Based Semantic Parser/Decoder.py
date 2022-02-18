## Original Code From https://github.com/tensorflow/docs/blob/master/site/en/tutorials/text/transformer.ipynb

##Copyright 2019 The TensorFlow Authors.
#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


## This code modifies the source code referenced above to create a Decoder-only mechanism with the choice of using feed-forward or recurrent weights for the attention and/or the hidden layer

import tensorflow_datasets as tfds
import tensorflow as tf
import logging
import time
import numpy as np
from sklearn.model_selection import train_test_split
import json
import tensorflow_text as tf_text
logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings


train_examples = []
val_examples = []
test_examples = []

# Select the dataset to be used for training
pathfile = "./Jobs640/Jobs640Train.txt"
if(".json" in pathfile):
    for line in open(pathfile,'r'):
        line = line.replace("_","~")
        if("tgt" in line):
            train_examples.append((tf.constant(",".join(json.loads(line)["src"]).replace(","," ")),tf.constant(",".join(json.loads(line)["tgt"]).replace(","," "))))
        elif("token" in line):
            train_examples.append((tf.constant(",".join(json.loads(line)["src"]).replace(","," ")),tf.constant(",".join(json.loads(line)["token"]).replace(","," "))))
else:
    for x in open(pathfile,'r'):
    ## Underscore not allowed in tokenizer, so replace it with "~" - will be reversed in testing
        x=x.replace("_","~")
        x=x.split("\t")
        train_examples.append((tf.constant(x[0]),tf.constant(x[1])))

# Select the dataset to be used for testing
pathfile = "./Jobs640/Jobs640Test.txt"
if(".json" in pathfile):
    for line in open(pathfile,'r'):
        line = line.replace("_","~")
        if("tgt" in line):
            test_examples.append((tf.constant(",".join(json.loads(line)["src"]).replace(","," ")),tf.constant(",".join(json.loads(line)["tgt"]).replace(","," "))))
        elif("token" in line):
            test_examples.append((tf.constant(",".join(json.loads(line)["src"]).replace(","," ")),tf.constant(",".join(json.loads(line)["token"]).replace(","," "))))
else:
    for x in open(pathfile,'r'):
        ## Underscore not allowed in tokenizer, so replace it with "~" - will be reversed in testing
        x=x.replace("_","~")
        x=x.split("\t")
        test_examples.append((tf.constant(x[0]),tf.constant(x[1])))

### Uncomment if there is a validation dataset
### Select the dataset to be used for training
##pathfile = ""
##if(".json" in pathfile):
##    for line in open(pathfile,'r'):
##        line = line.replace("_","~")
##        val_examples.append((tf.constant(",".join(json.loads(line)["src"]).replace(","," ")),tf.constant(",".join(json.loads(line)["tgt"]).replace(","," "))))
##else:
##    for x in open(pathfile,'r'):
##    ## Underscore not allowed in tokenizer, so replace it with "~" - will be reversed in testing
##        x=x.replace("_","~")
##        x=x.split("\t")
##        val_examples.append((tf.constant(x[0]),tf.constant(x[1])))
                

# Can be used if the dataset is not split yet.
# Set the test_size to define the percentage of the train-test split. Set shuffle to True or False
# according to your choice
##train_examples,test_examples=train_test_split(dataset, test_size=0.318, shuffle=True)

train=[]
train_target=[]
train_set=[]

test=[]
test_target=[]
test_set=[]

### To be used if there is a validation dataset
val=[]
val_target=[]
val_set=[]

for x in train_examples:
    train.append(x[0].numpy().decode("utf-8"))
    train_target.append(x[1].numpy().decode("utf-8"))
    train_set.append(x[0].numpy().decode("utf-8") + "\t" + x[1].numpy().decode("utf-8"))
for x in test_examples:
    test.append(x[0].numpy().decode("utf-8"))
    test_target.append(x[1].numpy().decode("utf-8"))
    test_set.append(x[0].numpy().decode("utf-8") + "\t" + x[1].numpy().decode("utf-8"))
    
### Uncomment if there is a validation dataset
##for x in val_examples:
##    va.append(x[0].numpy().decode("utf-8"))
##    test_target.append(x[1].numpy().decode("utf-8"))
##    test_set.append(x[0].numpy().decode("utf-8") + "\t" + x[1].numpy().decode("utf-8"))

train_examples1 = tf.data.Dataset.from_generator(
    lambda: train_examples, (tf.string, tf.string))

test_examples1 = tf.data.Dataset.from_generator(
    lambda: test_examples, (tf.string, tf.string))

### Uncomment if there is a validation set
##val_examples1 = tf.data.Dataset.from_generator(
##    lambda: val_examples, (tf.string, tf.string))

print("Size of train set:",len(train_set))
print("Size of test set:",len(test_set))
# Uncomment if there is a validation set
##print("Size of test set:",len(val_set))

# SubWord Tokenization 
# tokenizer_en - OUTPUT LANGUAGE
tokenizer_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
    (en.numpy() for pt, en in train_examples1), target_vocab_size=2**13)
# tokenizer_pt - INPUT LANGUAGE
tokenizer_pt = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
    (pt.numpy() for pt, en in train_examples1), target_vocab_size=2**13)

## Set True if you would like to see a sample tokenization
print_out = True

sample_string = test[0]

tokenized_string = tokenizer_pt.encode(sample_string)

original_string = tokenizer_pt.decode(tokenized_string)

assert original_string == sample_string

if(print_out):
    print ('Tokenized string is {}'.format(tokenized_string))
    print ('The original string: {}'.format(original_string))
    for ts in tokenized_string:
      print ('{} ----> {}'.format(ts, tokenizer_pt.decode([ts])))

sample_string = test_target[0]

tokenized_string = tokenizer_en.encode(sample_string)

original_string = tokenizer_en.decode(tokenized_string)

assert original_string == sample_string

if(print_out):
    print ('Tokenized string is {}'.format(tokenized_string))
    print ('The original string: {}'.format(original_string))
    for ts in tokenized_string:
      print ('{} ----> {}'.format(ts, tokenizer_en.decode([ts])))

BUFFER_SIZE = 20000

def encode(lang1, lang2):
  lang1 = [tokenizer_pt.vocab_size] + tokenizer_pt.encode(
      lang1.numpy()) + [tokenizer_pt.vocab_size+1]

  lang2 = [tokenizer_en.vocab_size] + tokenizer_en.encode(
      lang2.numpy()) + [tokenizer_en.vocab_size+1]
  
  return lang1, lang2

MAX_LENGTH = 40
BATCH_SIZE = 64

def filter_max_length(x, y, max_length=MAX_LENGTH):
  return tf.logical_and(tf.size(x) <= max_length,
                        tf.size(y) <= max_length)

def tf_encode(pt, en):
  return tf.py_function(encode, [pt, en], [tf.int64, tf.int64])

train_dataset = train_examples1.map(tf_encode)
train_dataset = train_dataset.filter(filter_max_length)
# cache the dataset to memory to get a speedup while reading from it.
train_dataset = train_dataset.cache()
train_dataset = train_dataset.padded_batch(
    BATCH_SIZE, padded_shapes=([-1], [-1]))
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

if(val_set!=[]):
    val_dataset = val_examples1.map(tf_encode)
    ### cache the dataset to memory to get a speedup while reading from it.
    val_dataset = val_dataset.cache()
    val_dataset = val_dataset.filter(filter_max_length).padded_batch(
        BATCH_SIZE, padded_shapes=([-1], [-1]))

def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates

def positional_encoding(position, d_model):
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)
  
  # apply sin to even indices in the array; 2i
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
  
  # apply cos to odd indices in the array; 2i+1
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

 
  pos_encoding = angle_rads[np.newaxis, ...]
    
  return tf.cast(pos_encoding, dtype=tf.float32)

def create_padding_mask(seq):
  seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
  # add extra dimensions to add the padding
  # to the attention logits.
  return seq[:, tf.newaxis, tf.newaxis, :] # (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(size):
  mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
  return mask  # (seq_len, seq_len)

def scaled_dot_product_attention(q, k, v, mask):
  """Calculate the attention weights.
  q, k, v must have matching leading dimensions.
  k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
  The mask has different shapes depending on its type(padding or look ahead) 
  but it must be broadcastable for addition.
  
  Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable 
          to (..., seq_len_q, seq_len_k). Defaults to None.
    
  Returns:
    output, attention_weights
  """
  
  # (batch_size, seq_len_q, seq_len_k, depth)
  matmul_qk = tf.matmul(q,k,transpose_b=True)

  # scale matmul_qk
  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

  if mask is not None:
      scaled_attention_logits += (mask * -1e9)
      
  # softmax is normalized on the last axis (seq_len_k) so that the scores
  # add up to 1.
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (batch_size, num_heads, seq_len_q, seq_len_k)

  output = tf.matmul(attention_weights, v) # (batch_size, num_heads, seq_len_q, depth)

  # softmax is normalized on the last axis (seq_len_k) so that the scores
  # add up to 1.
  scaled_attention_weights = tf.nn.softmax(scaled_attention_logits, axis=2)  # (batch_size, seq_len_q, seq_len_k, depth)

  return output, scaled_attention_weights

def print_out(q, k, v):
  temp_out, temp_attn = scaled_dot_product_attention(
      q, k, v, None)
  print ('Attention weights are:')
  print (temp_attn)
  print ('Output is:')
  print (temp_out)

class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model
    
    assert d_model % self.num_heads == 0
    
    self.depth = d_model // self.num_heads
    self.dense = tf.keras.layers.Dense(d_model)

    if(recurrent_attention==False):
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
    elif(recurrent_attention=="GRU"):
        self.wq = tf.keras.layers.GRU(d_model,return_sequences=True,return_state=True,recurrent_initializer='glorot_uniform')
        self.wk = tf.keras.layers.GRU(d_model,return_sequences=True,return_state=True,recurrent_initializer='glorot_uniform')
        self.wv = tf.keras.layers.GRU(d_model,return_sequences=True,return_state=True,recurrent_initializer='glorot_uniform')
    elif(recurrent_attention=="LSTM"):
        self.wq = tf.keras.layers.LSTM(d_model,return_sequences=True,return_state=True,recurrent_initializer='glorot_uniform')
        self.wk = tf.keras.layers.LSTM(d_model,return_sequences=True,return_state=True,recurrent_initializer='glorot_uniform')
        self.wv = tf.keras.layers.GRU(d_model,return_sequences=True,return_state=True,recurrent_initializer='glorot_uniform')
  ## To be used in training
  def split_heads(self, x, batch_size, vector):
    """Split the last dimension into (num_heads, depth).
    Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
    """
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])

  def call(self, v, k, q, mask, training):
    batch_size = tf.shape(q)[0]
    if(recurrent_attention==False):
        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)
    elif(recurrent_attention=="GRU"):
        q, q_state = self.wq(q)  # (batch_size, seq_len, d_model)
        k, k_state = self.wk(k)  # (batch_size, seq_len, d_model)
        v, v_state = self.wv(v)  # (batch_size, seq_len, d_model)
    elif(recurrent_attention=="LSTM"):
        q = self.wq(q)[0]  # (batch_size, seq_len, d_model)
        k = self.wk(k)[0]  # (batch_size, seq_len, d_model)
        v = self.wv(v)[0]  # (batch_size, seq_len, d_model)

    q_split = self.split_heads(q, batch_size, "query")  # (batch_size, num_heads, seq_len_q, depth)
    k_split = self.split_heads(k, batch_size, "key")  # (batch_size, num_heads, seq_len_k, depth)
    v_split = self.split_heads(v, batch_size, "value")  # (batch_size, num_heads, seq_len_k, depth)

    scaled_attention, attention_weights = scaled_dot_product_attention(
            q_split, k_split, v_split, mask)

    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
 
    concat_attention = tf.reshape(scaled_attention, 
                                  (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)
            
    output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
    
    return output, attention_weights

def point_wise_feed_forward_network(d_model, dff):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
  ])

def point_wise_feed_forward_network_recurrent_GRU(d_model,dff,dropout):
  return tf.keras.Sequential([
      tf.keras.layers.GRU(dff,return_sequences=True,return_state=False,recurrent_initializer='glorot_uniform',dropout=dropout),
      tf.keras.layers.GRU(d_model,return_sequences=False,return_state=False,recurrent_initializer='glorot_uniform',dropout=dropout)
  ])

def point_wise_feed_forward_network_recurrent_LSTM(d_model,dff,dropout):
  return tf.keras.Sequential([
      tf.keras.layers.LSTM(dff,return_sequences=True,return_state=False,recurrent_initializer='glorot_uniform',dropout=dropout),
      tf.keras.layers.LSTM(d_model,return_sequences=False,return_state=False,recurrent_initializer='glorot_uniform',dropout=dropout)
  ])

class DecoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(DecoderLayer, self).__init__()

    self.mha1 = MultiHeadAttention(d_model, num_heads)
    self.mha2 = MultiHeadAttention(d_model, num_heads)

    if(no_hidden==False):
        if(recurrent_hidden=="GRU"):
            self.rnn = point_wise_feed_forward_network_recurrent_GRU(d_model,dff,dropout=rate)
        elif(recurrent_hidden=="LSTM"):
            self.rnn = point_wise_feed_forward_network_recurrent_LSTM(d_model,dff,dropout=rate)
        else:
            self.ffn = point_wise_feed_forward_network(d_model,dff)
 
    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    
    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)
    self.dropout3 = tf.keras.layers.Dropout(rate)

  def call(self, x, dec_input, training, 
           look_ahead_mask, padding_mask):

    attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask, training=training)  # (batch_size, target_seq_len, d_model)
    attn1 = self.dropout1(attn1, training=training)
    out1 = self.layernorm1(attn1 + x)

    attn2, attn_weights_block2 = self.mha2(          # attn2 - (batch_size, target_seq_len, d_model) - This is the output of the attention layer upon concatenation
    dec_input, dec_input, out1, padding_mask, training=training)  # attn_weights_block2 - (batch_size, num_heads, seq_len_q, seq_len_k)
                                                 # target_seq_len = seq_len_q
    attn2 = self.dropout2(attn2, training=training)
    out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)
                                         # target_seq_len = seq_len_q
                                         
    if(no_hidden==True):
        pass
    else:
        if(recurrent_hidden=="GRU" or recurrent_hidden=="LSTM"):
            output = self.rnn(out2)[0]
        else:
            output = self.ffn(out2)
        out3 = self.dropout3(output, training=training)
        out3 = self.layernorm3(out3 + out2)  # (batch_size, target_seq_len, d_model)
                                                        
    return out2
    
class Decoder(tf.keras.layers.Layer):
  def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size,
               maximum_position_encoding, rate=0.1):
    super(Decoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers
    self.depth = int(self.d_model/num_heads)
    
    self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
    self.embedding2 = tf.keras.layers.Embedding(input_vocab_size, d_model)
    
    self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

    self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) 
                       for _ in range(num_layers)]
    
    self.dropout = tf.keras.layers.Dropout(rate)
    
  def call(self, x, dec_input, training, 
           look_ahead_mask, padding_mask):

    # adding embedding and position encoding for Decoder.
    seq_len = tf.shape(x)[1]
    decoder_attention_weights = {}

    x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x += self.pos_encoding[:, :seq_len, :]
    
    x = self.dropout(x, training=training)

    # adding embedding and position encoding for Decoder input
    seq_len = tf.shape(dec_input)[1]
    dec_input = self.embedding2(dec_input)  # (batch_size, input_seq_len, d_model)
    dec_input *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    dec_input += self.pos_encoding[:, :seq_len, :]

    dec_input = self.dropout(dec_input, training=training)
    
    for i in range(self.num_layers):
      x = self.dec_layers[i](x, dec_input, training,
                                             look_ahead_mask, padding_mask)
    return x

class Transformer(tf.keras.Model):
  def __init__(self, num_layers, d_model, num_heads_decoder, dff, input_vocab_size, 
               target_vocab_size, pe_input, pe_target, rate):
    super(Transformer, self).__init__()

    self.decoder = Decoder(num_layers, d_model, num_heads_decoder, dff, 
                           input_vocab_size, target_vocab_size, pe_target, rate)

    self.final_layer = tf.keras.layers.Dense(target_vocab_size)
    
  def call(self, inp, tar, training, dec_padding_mask1, 
           decoder_look_ahead_mask, dec_padding_mask2):

    dec_output = self.decoder(
            tar, inp, training, decoder_look_ahead_mask, dec_padding_mask2)

    final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

    return final_output

input_vocab_size = tokenizer_pt.vocab_size + 2
target_vocab_size = tokenizer_en.vocab_size + 2

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super(CustomSchedule, self).__init__()
    
    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps
    
  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)
    
    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

from tensorflow.keras.losses import categorical_crossentropy

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, name='sparse_categorical_crossentropy')

def scce_with_ls(y, y_hat):
    y = tf.one_hot(tf.cast(y, tf.int32), y_hat.shape[2])
    return categorical_crossentropy(y, y_hat, from_logits=True, label_smoothing = label_smoothing)

def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = scce_with_ls(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask
  
  return tf.reduce_mean(loss_)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='train_accuracy')

val_loss = tf.keras.metrics.Mean(name='val_loss')
val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='val_accuracy')

def create_masks(inp, tar):
  # Decoder padding mask 1.
  dec_padding_mask1 = create_padding_mask(inp)

  # Decoder padding mask 2. Used in the 2nd attention block in the decoder.
  # This padding mask is used to mask the decoder inputs.
  dec_padding_mask2 = create_padding_mask(inp)  

  look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
  dec_target_padding_mask2 = create_padding_mask(tar)
  combined_mask = tf.maximum(dec_target_padding_mask2, look_ahead_mask)
  
  return dec_padding_mask1, combined_mask, dec_padding_mask2


# The @tf.function trace-compiles train_step into a TF graph for faster
# execution. The function specializes to the precise shape of the argument
# tensors. To avoid re-tracing due to the variable sequence lengths or variable
# batch sizes (the last batch is smaller), use input_signature to specify
# more generic shapes.

train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]

@tf.function(input_signature=train_step_signature)
def train_step(inp, tar):
  tar_inp = tar[:, :-1]
  tar_real = tar[:, 1:]
  
  dec_padding_mask1, combined_mask, dec_padding_mask2 = create_masks(inp, tar_inp)
  
  with tf.GradientTape() as tape:
    predictions = transformer(inp, tar_inp, 
                                 True, 
                                 dec_padding_mask1, 
                                 combined_mask, 
                                 dec_padding_mask2)
    loss = loss_function(tar_real, predictions)
      
  gradients = tape.gradient(loss, transformer.trainable_variables)

  optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
  
  train_loss(loss)
  train_accuracy(tar_real, predictions)

@tf.function(input_signature=train_step_signature)
def val_step(inp, tar):
  tar_inp = tar[:, :-1]
  tar_real = tar[:, 1:]
  
  dec_padding_mask1, combined_mask, dec_padding_mask2 = create_masks(inp, tar_inp)
  
  with tf.GradientTape() as tape:
    predictions = transformer(inp, tar_inp, 
                                 False, 
                                 dec_padding_mask1, 
                                 combined_mask, 
                                 dec_padding_mask2)
    loss = loss_function(tar_real, predictions)
  
  val_loss(loss)
  val_accuracy(tar_real, predictions)

def evaluate(inp_sentence):
  start_token = [tokenizer_pt.vocab_size]
  end_token = [tokenizer_pt.vocab_size + 1]
  
  # adding the start and end token to the input sentence
  inp_sentence = start_token + tokenizer_pt.encode(inp_sentence) + end_token
  decoder_input1 = tf.expand_dims(inp_sentence, 0)

  # adding a start token to the target beginning of the target sentence
  decoder_input2 = [tokenizer_en.vocab_size]
  output = tf.expand_dims(decoder_input2, 0)
    
  for i in range(MAX_LENGTH):
    dec_padding_mask1, combined_mask, dec_padding_mask2 = create_masks(
        decoder_input1, output)

    # predictions.shape == (batch_size, seq_len, vocab_size)
    predictions = transformer(decoder_input1, 
                                                 output,
                                                 False,
                                                 dec_padding_mask1,
                                                 combined_mask,
                                                 dec_padding_mask2)              
    
    # select the last word from the seq_len dimension
    predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)

    predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
    
    # return the result if the predicted_id is equal to the end token
    if predicted_id == tokenizer_en.vocab_size+1:
        return tf.squeeze(output, axis=0)
    
    # concatentate the predicted_id to the output which is given to the decoder
    # as its input.
    output = tf.concat([output, predicted_id], axis=-1)

  return tf.squeeze(output, axis=0)

def plot_attention_weights(attention, sentence, result, layer):
  fig = plt.figure(figsize=(16, 8))
  
  sentence = tokenizer_pt.encode(sentence)
  attention = tf.squeeze(attention[layer], axis=0)

  for head in range(attention.shape[0]):
    ax = fig.add_subplot(2, 4, head+1)
    
    # plot the attention weights
    ax.matshow(attention[head][:, :], cmap='viridis')
    fontdict = {'fontsize': 10}

    ax.set_xticks(range(len(sentence)+2))
    ax.set_yticks(range(len(result)))
    
    ax.set_ylim(len(result)-0.5, -0.5)

    ax.set_xticklabels(
        ['[START]']+[tokenizer_pt.decode([i]) for i in sentence]+['[END]'], 
        fontdict=fontdict, rotation=90)
    ax.set_yticklabels([tokenizer_en.decode([i]) for i in result
                        if i < tokenizer_en.vocab_size]+['[END]'], 
                       fontdict=fontdict)
    
    ax.set_xlabel('Head {}'.format(head+1))
  
  plt.tight_layout()
  plt.show()


def translate(sentence, plot=''):

  result = evaluate(sentence)

  predicted_sentence = tokenizer_en.decode([i for i in result 
                                            if i < tokenizer_en.vocab_size])  
        
  return sentence.strip() + "\t" + predicted_sentence.strip()


# Hyperparameters
# Dimension of the word embeddings
dimension = 512
# Number of units in the hidden layers
num_units = 128
# Number of layers for the decoder
num_layers = 1
# Number of attention heads for the decoder
num_attention_heads_decoder = 8
# Dropout value for the optimizer
dropout_rate = 0.1
# Drouput value for the optimizer
label_smoothing = 0.0
# Set "GRU" or "LSTM" for recurrent attention weights. Else set False
recurrent_attention = False
# Set "GRU"' or "LSTM" for recurrent attention weights. Else set False
recurrent_hidden = False
# Set True if you would like to have a hidden layer. Else set False
no_hidden = False
# Number of Epochs for the training
EPOCHS = 100

tf.config.run_functions_eagerly(True)

learning_rate = CustomSchedule(dimension)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, 
                                         epsilon=1e-9)

 
transformer = Transformer(num_layers, dimension, num_attention_heads_decoder, num_units,
                  input_vocab_size=input_vocab_size,
                  target_vocab_size=target_vocab_size, 
                  pe_input=1000, 
                  pe_target=1000,
                  rate=dropout_rate)

checkpoint_path = "./checkpoints"
  
ckpt = tf.train.Checkpoint(transformer=transformer,
                           optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
# if a checkpoint exists, restore the latest checkpoint
if ckpt_manager.latest_checkpoint:
  ckpt.restore(ckpt_manager.latest_checkpoint)
  print ('Latest checkpoint restored!!')

for epoch in range(0,EPOCHS):
    print("EPOCH:",epoch+1)
    start = time.time()
  
    train_loss.reset_states()
    train_accuracy.reset_states()

    val_loss.reset_states()
    val_accuracy.reset_states()
  
    for (batch, (inp, tar)) in enumerate(train_dataset):
        train_step(inp, tar)

    # Print out report every 50 epochs. Can be changes by replacing 50 in the if statement
    if((epoch+1)%50==0):
        print()
        print ('Epoch {} Training Loss {:.4f} Training Accuracy {:.4f}'.format(epoch + 1, 
                                                    train_loss.result(), 
                                                    train_accuracy.result()))

        print ('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

        ckpt_save_path = ckpt_manager.save()
        print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                             ckpt_save_path))


# Testing and accuracy report
start = time.time()
start_token = [tokenizer_pt.vocab_size]
end_token = [tokenizer_pt.vocab_size + 1]

answers=[]
count=0
for x in range(0,len(test)):
    input_sentence = [start_token + tokenizer_pt.encode(test[x]) + end_token]
    a=translate(test[x]).replace("~","_")
    answers.append(a)

right_answers=[]
wrong_answers=[]
for y in test_set:
    cnt=0
    for x in answers:
        question=x.split("\t")[0]
        y=y.replace("~","_")
        if(x.strip() == y.strip()):
            right_answers.append(x.strip())
            count+=1
            cnt+=1
            break

    if(cnt==0):
        z1=y.strip().split("\t")[0].strip()
        for x in answers:
            z2=x.strip().split("\t")[0].strip()
            if(z1==z2):
                wrong_answers.append("Prediction: "+x.strip()+"\t"+" Right Parse: "+y.strip().split("\t")[1].strip())
                break                                 
                                                                               
print("No. of correct parses:",len(right_answers))
print("No. of wrong parses:",len(wrong_answers))
print("No. of total parses:", len(test_set))
print("Score:",round(count/len(test_set),2)*100)
print ('Time taken for evaluation {} secs\n'.format(time.time() - start))

# Output the results in .txt format
s = open("./TestReport.txt","w")
s.write("No. of correct parses: "+str(len(right_answers))+"\n")
s.write("No. of wrong parses: "+str(len(wrong_answers))+"\n")
s.write("No. of total parses: "+str(len(test_set))+"\n")
s.write("Score:"+str(round(count/len(test_set),2)*100)+"\n")
s.write('Time taken for evaluation {} secs\n'.format(time.time() - start)+"\n")
s.write("Correct Parses:"+"\n")
for x in right_answers:
    s.write(x.strip()+"\n")
s.write("\n"+"Wrong Parses:"+"\n")
for x in wrong_answers:
    s.write(x.strip()+"\n")
s.close()

# Output the shuffled train and test sets in .txt format
s = open("./ShuffledTrainSet.txt","w")
for x in train_set:
    s.write(x.strip()+"\n")
s.close()
s = open("./ShuffledTestSet.txt","w")
for x in test_set:
    s.write(x.strip()+"\n")
s.close()
            
                            


# Named Entity Recognition (NER) with TensorFlow

Slides: https://docs.google.com/presentation/d/1eUEOTSeUnR2Sz1uDF4e3YvBaxQ1kLUjog9qkhgBEMV0/edit?usp=sharing


```python
import tensorflow as tf
import pandas as pd
import numpy as np
import csv
```

### Data Preparation

Data is from here https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus/data, download ner_dataset.csv from the ZIP archive.




```python
validation_sentence = 'While speaking on Channels Television on Thursday April 5 2018 Adesina said the fund is not just to intensify the military fight against Boko Haram but to fight other forms of insecurity in the country'

validation_tags = ['O', 'O', 'O', 'B-ORG', 'I-ORG', 'O', 'B-TIM', 'I-TIM', 'I-TIM', 'I-TIM', 'B-PER', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
                  'O', 'B-ORG', 'I-ORG', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
```

Below we parse the file to load sentences and tags into different lists. Also, we only want sentences not more than 35 words long (same length as our validation sentence above)


```python
sentences = []
tags = []
max_length = len(validation_tags)

with open('data/ner_dataset.csv', 'rb') as csvfile:
    ner_data = csv.reader(csvfile, delimiter=',')
    sentence = []
    tag = []
    for row in ner_data:
        
        sentence.append(row[1])
        tag.append(row[3].upper())
        
        if row[1] == '.':
            if len(sentence) <= max_length:
                sentences.append(sentence)
                tags.append(tag)
            sentence = []
            tag = []
```

Below is sample entries of `sentences` and `tags`


```python
print sentences[:2]
print
print tags[:2]
```

    [['Word', 'Thousands', 'of', 'demonstrators', 'have', 'marched', 'through', 'London', 'to', 'protest', 'the', 'war', 'in', 'Iraq', 'and', 'demand', 'the', 'withdrawal', 'of', 'British', 'troops', 'from', 'that', 'country', '.'], ['Families', 'of', 'soldiers', 'killed', 'in', 'the', 'conflict', 'joined', 'the', 'protesters', 'who', 'carried', 'banners', 'with', 'such', 'slogans', 'as', '"', 'Bush', 'Number', 'One', 'Terrorist', '"', 'and', '"', 'Stop', 'the', 'Bombings', '.']]
    
    [['TAG', 'O', 'O', 'O', 'O', 'O', 'O', 'B-GEO', 'O', 'O', 'O', 'O', 'O', 'B-GEO', 'O', 'O', 'O', 'O', 'O', 'B-GPE', 'O', 'O', 'O', 'O', 'O'], ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-PER', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']]


We'll need to create a vocabulary from our sentences i.e a set of unique words. We'll do same for the tags too


```python
unique_tags = list(set(t for tagset in tags for t in tagset))
vocabulary = list(set(word for sentence in sentences for word in sentence))
```


```python
print unique_tags
```

    ['I-ART', 'I-EVE', 'B-EVE', 'B-GPE', 'B-TIM', 'I-TIM', 'B-ORG', 'B-ART', 'O', 'B-GEO', 'I-GPE', 'TAG', 'I-GEO', 'B-PER', 'I-PER', 'I-ORG', 'I-NAT', 'B-NAT']



```python
print vocabulary[:10]
print 'Number of words in vocabulary', len(vocabulary)
```

    ['heavily-fortified', 'mid-week', '1,800', 'Pronk', 'woods', 'Safarova', 'Nampo', 'hanging', 'trawling', 'five-nation']
    Number of words in vocabulary 33105



```python
train_sentences = sentences[:int(.7 * len(sentences))]
train_tags = tags[:int(.7 * len(tags))]

test_sentences = sentences[int(.7 * len(tags) + 1):]
test_tags = tags[int(.7 * len(tags) + 1):]
```


```python
len(train_sentences), len(test_sentences), len(sentences)
```




    (31732, 13599, 45332)



### Model Architecture
Simple LSTM network with a softmax at the end

Important NOTE: If you want to run the network using a one-hot encoding of the words, make sure `batch_size` is set to something low. Higher values might result in your computer freezing. I tried on my core i5, 8GB RAM laptop and it wasn't pleasant. So stick with default value of 8 for batch_size or lower.


```python
# Parameters
learning_rate = 0.001
batch_size = 8
target_size = len(unique_tags)
display_size = 50

# Network Parameters
n_features = len(vocabulary)
sequence_length = 10
n_hidden = 128 # hidden layer num of features

tf.reset_default_graph()

# tf Graph input
X = tf.placeholder('float', [None, max_length, n_features], name='X')
Y = tf.placeholder('float', [None, max_length, target_size], name='Y')

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, target_size]))
}
biases = {
    'out': tf.Variable(tf.random_normal([target_size]))
}
```


```python

cell = tf.contrib.rnn.LSTMCell(n_hidden, state_is_tuple=True)

output, _ = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

output = tf.reshape(output, [-1, n_hidden])

prediction = tf.matmul(output, weights['out']) + biases['out']
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=Y))

minimize = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)
```


```python
init = tf.global_variables_initializer()
num_batches = int(len(train_sentences)) / batch_size
epoch = 1
print 'Number of batches:', num_batches
```

    Number of batches: 469



```python
len(train_sentences)
```




    15024



### Run graph using one-hot encoding of words


```python
with tf.Session() as sess:
    sess.run(init)
    for i in range(epoch):        
        
        for j in range(num_batches):
            ptr = 0
            batch_X = []
            batch_Y = []
            for _ in range(batch_size):
                x, y = (train_sentences[ptr: ptr + 1], 
                        train_tags[ptr: ptr + 1])            

                x_one_hot = []

                for s in x[0]:
                    x_one_hot.append(np.eye(len(vocabulary))[vocabulary.index(s)])
                    
                for remainder in range(max_length - len(x_one_hot)):
                    x_one_hot.append([0]*len(vocabulary))
                    
                batch_X.append(x_one_hot)              

                y_one_hot = []

                for t in y[0]:
                    y_one_hot.append(np.eye(target_size)[unique_tags.index(t)])
                    
                for remainder in range(max_length - len(y_one_hot)):
                    y_one_hot.append(np.eye(target_size)[unique_tags.index('O')])
                    
                batch_Y.append(y_one_hot)

                ptr += 1
            
            _, entropy, preds = sess.run([minimize, cross_entropy, prediction],{X: np.array(batch_X).reshape(batch_size, max_length, len(vocabulary)), Y: np.array(batch_Y).reshape(batch_size, max_length, target_size)})
            
            if j % display_size == 0:
                print 'Loss at batch {0}'.format(j), entropy

        print "Epoch ",str(i)
    
```

    Loss at batch 0 87.09978
    Loss at batch 50 9.664997


### Word Embeddings
We'll use Google's word2vec which you can grab from here https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit.
To load the word embeddings, we'll neeed another tool, `gensim`.


```python
from gensim.models import word2vec, KeyedVectors
```

Load the word vectors like so. This operations takes a good while on my laptop; core i5.


```python
w2v = KeyedVectors.load_word2vec_format('path/to/word2vec', binary=True)
```

Below is what `boy` is represented according to the embedding


```python
w2v.word_vec('boy')
```




    array([ 2.35351562e-01,  1.65039062e-01,  9.32617188e-02, -1.28906250e-01,
            1.59912109e-02,  3.61328125e-02, -1.16699219e-01, -7.32421875e-02,
            1.38671875e-01,  1.15356445e-02,  1.87500000e-01, -2.91015625e-01,
            1.70898438e-02, -1.84570312e-01, -2.87109375e-01,  2.54821777e-03,
           -2.19726562e-01,  1.77734375e-01, -1.20605469e-01,  5.39550781e-02,
            3.78417969e-02,  2.49023438e-01,  1.76757812e-01,  2.69775391e-02,
            1.21093750e-01, -3.51562500e-01, -5.83496094e-02,  1.22070312e-01,
            5.97656250e-01, -1.60156250e-01,  1.08398438e-01, -2.40478516e-02,
           -1.16699219e-01,  3.58886719e-02, -2.37304688e-01,  1.15234375e-01,
            5.27343750e-01, -2.18750000e-01, -4.54101562e-02,  3.30078125e-01,
            3.75976562e-02, -5.51757812e-02,  3.26171875e-01,  6.74438477e-03,
            3.71093750e-01,  3.68652344e-02,  6.68945312e-02,  5.17578125e-02,
           -4.76074219e-02, -7.91015625e-02,  4.46777344e-02,  1.67968750e-01,
            5.51757812e-02, -2.91015625e-01,  2.59765625e-01, -1.00097656e-01,
           -1.09863281e-01, -9.15527344e-03,  2.63671875e-02, -3.44238281e-02,
            9.37500000e-02,  3.53515625e-01,  8.39843750e-02, -7.75146484e-03,
            8.64257812e-02, -5.24902344e-02, -5.59082031e-02, -8.59375000e-02,
            5.37109375e-02, -1.47094727e-02,  3.63769531e-02,  4.68750000e-02,
           -3.39843750e-01,  1.28906250e-01, -1.22558594e-01,  4.57031250e-01,
            1.27929688e-01, -2.89062500e-01,  1.56250000e-01,  3.73535156e-02,
            2.75390625e-01, -1.28784180e-02, -1.50390625e-01, -1.64062500e-01,
           -3.39843750e-01,  8.00781250e-02, -9.21630859e-03,  2.78320312e-02,
            9.32617188e-02,  2.25830078e-02, -1.62353516e-02, -8.25195312e-02,
           -1.90429688e-02, -3.49121094e-02,  9.42382812e-02,  3.66210938e-02,
            6.39648438e-02,  2.00195312e-01, -4.05273438e-02, -1.08886719e-01,
           -3.93676758e-03, -2.55859375e-01,  6.78710938e-02, -1.89453125e-01,
            1.72851562e-01, -1.73828125e-01,  2.07031250e-01, -1.59179688e-01,
            2.85339355e-03, -1.80664062e-01, -6.93359375e-02,  2.05078125e-01,
            5.93261719e-02, -2.17773438e-01, -1.36718750e-01, -4.91333008e-03,
           -1.38671875e-01, -7.47070312e-02, -3.54003906e-02,  1.13769531e-01,
            3.07617188e-02, -1.05957031e-01, -3.30078125e-01, -2.72216797e-02,
           -1.94091797e-02,  9.52148438e-02,  8.69140625e-02, -2.16064453e-02,
           -6.98242188e-02, -1.73828125e-01, -1.60156250e-01, -2.44140625e-01,
            9.82666016e-03,  2.24609375e-02, -2.13867188e-01,  1.91406250e-01,
            2.01171875e-01,  2.72216797e-02,  2.81982422e-02,  2.42187500e-01,
            3.55468750e-01, -5.32226562e-02,  1.78710938e-01,  6.78710938e-02,
           -6.73828125e-02,  3.49609375e-01, -1.92382812e-01, -1.00097656e-02,
           -2.05078125e-01, -1.59179688e-01,  3.76953125e-01, -2.15820312e-01,
           -2.36328125e-01,  6.49414062e-02, -1.39770508e-02,  4.22363281e-02,
            2.51464844e-02, -1.00585938e-01,  1.37695312e-01, -2.43164062e-01,
            1.20605469e-01,  2.03857422e-02,  3.12500000e-01,  1.09863281e-01,
           -1.04980469e-01, -9.13085938e-02,  2.21679688e-01, -1.04003906e-01,
            1.25976562e-01,  5.10253906e-02,  6.39648438e-02, -1.15722656e-01,
           -3.19824219e-02, -8.34960938e-02, -1.97265625e-01, -2.33154297e-02,
            1.94335938e-01,  2.24609375e-01, -2.30468750e-01,  4.17480469e-02,
            6.49414062e-02, -1.70898438e-01,  7.86132812e-02, -3.58886719e-02,
           -1.66015625e-01,  2.25585938e-01,  1.23535156e-01,  1.08398438e-01,
            1.15722656e-01,  7.37304688e-02, -1.56250000e-02, -5.85937500e-02,
           -8.93554688e-02,  1.30859375e-01,  1.90429688e-01, -3.58886719e-02,
           -1.36718750e-02, -1.88476562e-01, -1.48437500e-01, -2.51953125e-01,
           -1.22558594e-01, -2.75390625e-01, -1.54296875e-01, -2.83203125e-01,
            1.10839844e-01, -2.46093750e-01,  1.89453125e-01, -2.50244141e-02,
            8.59375000e-02, -1.17675781e-01, -2.46582031e-02, -1.32812500e-01,
            1.00097656e-01, -2.45117188e-01, -2.02148438e-01, -7.56835938e-02,
            6.03027344e-02,  1.72851562e-01, -6.59179688e-02,  6.78710938e-02,
            6.98242188e-02, -4.10156250e-02,  2.14843750e-01,  7.17773438e-02,
           -4.57763672e-03, -4.04357910e-04,  8.59375000e-02, -2.55859375e-01,
           -4.32128906e-02, -1.31835938e-01,  2.05078125e-02, -2.46093750e-01,
           -1.28906250e-01,  1.23535156e-01, -1.48437500e-01,  5.15136719e-02,
           -1.55273438e-01, -1.70898438e-01,  1.92382812e-01,  2.16796875e-01,
            5.81054688e-02, -1.28906250e-01, -1.43554688e-01, -7.78198242e-03,
           -1.15234375e-01,  4.08203125e-01, -3.37890625e-01,  8.64257812e-02,
            2.08007812e-01,  2.35595703e-02,  1.36718750e-01, -4.71191406e-02,
            9.91210938e-02,  1.18164062e-01,  1.19140625e-01,  1.24511719e-01,
            4.66308594e-02,  5.41992188e-02, -2.11914062e-01, -8.20312500e-02,
           -5.17578125e-02,  2.03857422e-02, -1.59179688e-01, -1.76757812e-01,
            8.54492188e-02,  1.38671875e-01, -1.01562500e-01,  2.61230469e-02,
           -1.88476562e-01, -1.57470703e-02,  1.21093750e-01, -9.66796875e-02,
            2.13623047e-02, -6.68945312e-02, -2.69775391e-02,  3.51562500e-02,
            1.68945312e-01,  1.55639648e-02, -1.25976562e-01, -1.44531250e-01,
            1.78710938e-01, -7.42187500e-02,  2.72216797e-02,  4.98046875e-01,
           -6.03027344e-02, -1.35742188e-01, -1.62109375e-01,  9.57031250e-02,
           -1.84326172e-02,  3.90625000e-01,  1.90429688e-02, -1.03149414e-02,
           -1.15234375e-01, -2.91015625e-01, -5.95703125e-02, -5.37109375e-02,
           -7.42187500e-02, -2.65625000e-01, -1.03027344e-01,  1.35742188e-01],
          dtype=float32)



### Run graph with words represented as word2vec
Same as architecture as pervious except `n_features` is now the dimension of the vector returned by word2vec


```python
# Parameters
learning_rate = 0.001
batch_size = 32
target_size = len(unique_tags)
display_size = 50

# Network Parameters
n_features = 300 # dimension of the vector return by word2vec
sequence_length = max_length
n_hidden = 512 # hidden layer num of features

tf.reset_default_graph()

# tf Graph input
X = tf.placeholder('float', [None, max_length, n_features], name='X')
Y = tf.placeholder('float', [None, max_length, target_size], name='Y')

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, target_size]))
}
biases = {
    'out': tf.Variable(tf.random_normal([target_size]))
}
```


```python

cell = tf.contrib.rnn.LSTMCell(n_hidden, state_is_tuple=True)

output, _ = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

output = tf.reshape(output, [-1, n_hidden])

prediction = tf.matmul(output, weights['out']) + biases['out']
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=Y))

minimize = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

```


```python
init = tf.global_variables_initializer()
num_batches = int(len(train_sentences)) / batch_size
epoch = 1
print 'Number of batches:', num_batches
```

    Number of batches: 991



```python
with tf.Session() as sess:
    sess.run(init)
    for i in range(epoch):        
        
        for j in range(num_batches):
            ptr = 0
            batch_X = []
            batch_Y = []
            for _ in range(batch_size):
                x, y = (train_sentences[ptr: ptr + 1], 
                        train_tags[ptr: ptr + 1])            

                x_one_hot = []

                for s in x[0]:
                    try:
                        x_one_hot.append(w2v.word_vec(s))
                    except:
                        #if word isn't in the word2vec, use zeroes
                        x_one_hot.append([0]*n_features)
                    
                for remainder in range(max_length - len(x_one_hot)):
                    #pad sentence remainder with zeroes
                    x_one_hot.append([0]*n_features)
                    
                batch_X.append(x_one_hot)              

                y_one_hot = []

                for t in y[0]:
                    y_one_hot.append(np.eye(target_size)[unique_tags.index(t)])
                    
                for remainder in range(max_length - len(y_one_hot)):
                    y_one_hot.append(np.eye(target_size)[unique_tags.index('O')])
                    
                batch_Y.append(y_one_hot)

                ptr += 1
            
            _, entropy, preds = sess.run([minimize, cross_entropy, prediction],{X: np.array(batch_X).reshape(batch_size, max_length, n_features), Y: np.array(batch_Y).reshape(batch_size, max_length, target_size)})
            
            if j % display_size == 0:
                print 'Loss at batch {0}'.format(j), entropy

        print "Epoch ",str(i)
    
```

    Loss at batch 0 4.4722333
    Loss at batch 50 0.015087938
    Loss at batch 100 0.0060136113
    Loss at batch 150 0.004677868
    Loss at batch 200 0.004170066
    Loss at batch 250 0.003910939
    Loss at batch 300 0.003752762
    Loss at batch 350 0.003642353
    Loss at batch 400 0.0035570299
    Loss at batch 450 0.0034858019
    Loss at batch 500 0.0034227856
    Loss at batch 550 0.0033645693
    Loss at batch 600 0.003309061
    Loss at batch 650 0.0032548874
    Loss at batch 700 0.00320115
    Loss at batch 750 0.0031472098
    Loss at batch 800 0.003092601
    Loss at batch 850 0.003036987
    Loss at batch 900 0.00298011
    Loss at batch 950 0.0029217617
    Epoch  0


Obvious benefit of using word2vec is that the network runs faster, converges quicker too. Runs faster because we've reduced the feature representation from an outrageous dimension in the length of the vocabulary (thousands) to only 300, the dimension of the array returned by word2vec.

### Prediction


```python
with tf.Session() as sess:
    sess.run(init)

    valid_X = []
    
    for word in validation_sentence.split(' '):
        try:
            valid_X.append(w2v.word_vec(word))
        except:
            #if word isn't in the word2vec, use zeroes
            valid_X.append([0]*n_features)       

    valid_Y = []

    for t in validation_tags:
        valid_Y.append(np.eye(target_size)[unique_tags.index(t)])
            
    preds = sess.run([prediction],{X: np.array(valid_X).reshape(1, max_length, n_features), Y: np.array(valid_Y).reshape(1, max_length, target_size)})

    valid_words = validation_sentence.split(' ')
    preds = np.array(preds).reshape(35, 18)

    for i, p in enumerate(preds):
        print 'Network predicted {0} for word {1}'.format(unique_tags[np.argmax(p)], valid_words[i])
```

    Network predicted B-GEO for word While
    Network predicted I-GEO for word speaking
    Network predicted I-GEO for word on
    Network predicted I-GEO for word Channels
    Network predicted I-GEO for word Television
    Network predicted I-GEO for word on
    Network predicted I-GEO for word Thursday
    Network predicted I-GEO for word April
    Network predicted B-GEO for word 5
    Network predicted B-GEO for word 2018
    Network predicted B-GEO for word Adesina
    Network predicted B-GEO for word said
    Network predicted B-GEO for word the
    Network predicted I-NAT for word fund
    Network predicted B-GEO for word is
    Network predicted B-GEO for word not
    Network predicted B-GEO for word just
    Network predicted B-GEO for word to
    Network predicted I-GEO for word intensify
    Network predicted I-GEO for word the
    Network predicted B-GEO for word military
    Network predicted B-GEO for word fight
    Network predicted B-GEO for word against
    Network predicted I-PER for word Boko
    Network predicted B-GEO for word Haram
    Network predicted B-GEO for word but
    Network predicted B-GEO for word to
    Network predicted B-GEO for word fight
    Network predicted I-PER for word other
    Network predicted B-GEO for word forms
    Network predicted B-GEO for word of
    Network predicted B-GEO for word insecurity
    Network predicted B-GEO for word in
    Network predicted B-GEO for word the
    Network predicted I-NAT for word country


### Things to try
- Use a bidirectional LSTM
- Add dropout
- Replace softmax with Linear-Chain CRF
- Try other word representations; Glove?
- Tune batch size, learning rate
- Add MOAR layers!!!
- Use longer sentences


More importantly, train on a better dataset. Like I mentioned, NER is domain specific. Our validation sentence contains details perhaps specific to Nigeria:
    - the name Adesina and
    - Channels Television

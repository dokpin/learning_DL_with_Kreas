from keras.datasets import imdb
import numpy as np

def vectorize_sequence(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

#print("train_data[0]")
#print(train_data[0])
#print("train_labels[0]")
#print(train_labels[0])

#print("max([max(sequence) for sequence in train_data])")
#print(max([max(sequence) for sequence in train_data]))

#print([max(sequence) for sequence in train_data])

# 데이터 디코딩
#word_index = imdb.get_word_index()
#reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
#decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])
#print(decoded_review)

# 정수 시퀀스 인코딩
x_train = vectorize_sequence(train_data)
x_test = vectorize_sequence(test_data)
#print("train_data[0] vector")
#print(x_train[0])

# 레이블 인코딩
y_train = np.asarray(train_labels).astype("float32")
y_test = np.asarray(test_labels).astype("float32")
#print("y_train[0] vector")
#print(y_train[0])

from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
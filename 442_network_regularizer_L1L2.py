from keras.datasets import imdb
import numpy as np
import matplotlib.pyplot as plt

def vectorize_sequence(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

# save np.load
np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

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

x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

origin_history = model.fit(partial_x_train, partial_y_train,
							 epochs=20, batch_size=512, validation_data=(x_val, y_val))

origin_history_dict = origin_history.history
origin_val_loss = origin_history_dict['val_loss']

###
from keras import regularizers

model2 = models.Sequential()
model2.add(layers.Dense(16, kernel_regularizer=regularizers.l1_l2(0.001),
                        activation='relu', input_shape=(10000,)))
model2.add(layers.Dense(16, kernel_regularizer=regularizers.l1_l2(0.001), activation='relu'))
model2.add(layers.Dense(1, activation='sigmoid'))

model2.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

L1L2regularized_history = model2.fit(partial_x_train, partial_y_train,
							 epochs=20, batch_size=512, validation_data=(x_val, y_val))

L1L2regularized_history_dict = L1L2regularized_history.history
L1L2regularized_val_loss = L1L2regularized_history_dict['val_loss']






epochs = range(1, len(L1L2regularized_val_loss) + 1)

plt.plot(epochs, origin_val_loss, 'P', label='Original model')
plt.plot(epochs, L1L2regularized_val_loss, 'bo', label='L1L2-regularized model')

plt.xlabel('Epochs')
plt.ylabel('Validation loss')
plt.legend()

plt.show()
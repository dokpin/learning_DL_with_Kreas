from keras.datasets import reuters
import numpy as np
import matplotlib.pyplot as plt

def vectorize_sequence(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results

# save np.load
np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

#print("len(train_data)")
#print(len(train_data))
#print("len(test_data)")
#print(len(test_data))

# 데이터 디코딩
#word_index = reuters.get_word_index()
#reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
#decoded_newswire = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])
#print(decoded_newswire)

#print('train_labels[0]')
#print(train_labels[0])

# 정수 시퀀스 인코딩
x_train = vectorize_sequence(train_data)
x_test = vectorize_sequence(test_data)
print("train_data[0] vector")
print(x_train[0])

# 레이블 인코딩
one_hot_train_labels = vectorize_sequence(train_labels)
one_hot_test_labels = vectorize_sequence(test_labels)
print("one_hot_train_labels[0] vector")
print(one_hot_train_labels[0])
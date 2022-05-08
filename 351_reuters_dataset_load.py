from keras.datasets import reuters
import numpy as np
import matplotlib.pyplot as plt

# save np.load
np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

print("len(train_data)")
print(len(train_data))
print("len(test_data)")
print(len(test_data))

# 데이터 디코딩
word_index = reuters.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
decoded_newswire = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])
print(decoded_newswire)

print('train_labels[0]')
print(train_labels[0])
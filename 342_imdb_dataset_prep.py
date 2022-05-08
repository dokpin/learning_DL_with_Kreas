from keras.datasets import imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

#print("train_data[0]")
#print(train_data[0])
#print("train_labels[0]")
#print(train_labels[0])

#print("max([max(sequence) for sequence in train_data])")
#print(max([max(sequence) for sequence in train_data]))

#print([max(sequence) for sequence in train_data])

word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])
print(decoded_review)
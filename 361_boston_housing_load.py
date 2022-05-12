from keras.datasets import boston_housing

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

print('train_data.shape')
print(train_data.shape)

print('test_data.shape')
print(test_data.shape)

print('train_targets.shape')
print(train_targets.shape)

print('train_targets')
print(train_targets)
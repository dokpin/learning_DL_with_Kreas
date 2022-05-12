from keras.datasets import boston_housing

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

print('train_data[0]')
print(train_data[0])

#print('train_data.mean(axis=1)')
#print(train_data.mean(axis=1))

mean = train_data.mean(axis=0)
print('mean')
print(mean)
train_data -= mean
std = train_data.std(axis=0)
print('std')
print(std)
train_data /= std

print('train_data[0]')
print(train_data[0])

test_data -= mean
test_data /= std
from keras.datasets import boston_housing

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std

from keras import models
from keras import layers

def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model
    
import numpy as np

k = 4

num_val_samples = len(train_data) // k
# 학습 데이터가 404개
print('num_val_samples')
print(num_val_samples)
num_epochs = 100
all_scores = []


for i in range(k):
    print('처리중인 폴드 #', i)
    print('i * num_val_samples')
    print(i * num_val_samples)
    print('(i + 1) * num_val_samples')
    print((i + 1) * num_val_samples)
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]

    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
        train_data[(i + 1) * num_val_samples:]],
        axis=0
    )

    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
        train_targets[(i + 1) * num_val_samples:]],
        axis=0
    )

    model = build_model()
    model.fit(partial_train_data, partial_train_targets,
                epochs=num_epochs, batch_size=1, verbose=0)
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)

print('all_scores')
print(all_scores)

print('np.mean(all_scores)')
print(np.mean(all_scores))
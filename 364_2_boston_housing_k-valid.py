from keras.datasets import boston_housing

#from tensorflow.keras.datasets import boston_housing
#from tensorflow.keras import Sequential, models, layers
#from tensorflow.keras.layers import Conv2D, Flatten, Dense

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
#print('num_val_samples')
#print(num_val_samples)
num_epochs = 500
all_mae_histories = []


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
    history = model.fit(partial_train_data, partial_train_targets,
                epochs=num_epochs, batch_size=1, verbose=1, validation_data=(val_data, val_targets))
    #print(history.history.items())
    mae_history = history.history['val_mean_absolute_error']
    #mae_history = history.history['mean_absolute_error']
    #mae_history = history.history['val_mae']
    all_mae_histories.append(mae_history)

average_mae_history = [
    np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

print('average_mae_history')
print(average_mae_history)

import matplotlib.pyplot as plt

plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

def smooth_curve(points, factor = 0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    
    return smoothed_points

smooth_mae_history = smooth_curve(average_mae_history[10:])

plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()



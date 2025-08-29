import pandas as pd

data = pd.read_csv('cdapple/gpascore.csv')

data = data.dropna()
ydata = data['admit'].values
xdata = []

for i, rows in data.iterrows():
    xdata.append([ rows['gre'], rows['gpa'], rows['rank'] ])

import numpy as np
import tensorflow as tf

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='tanh'),
    tf.keras.layers.Dense(128, activation='tanh'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(np.array(xdata), np.array(ydata), epochs=1000)

#예측
predict = model.predict([[750, 3.70, 3],[400, 2.2, 1]])
print(predict)
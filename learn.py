#Python program to train the data-sets for classification into 'coding' and 'non-coding' region.

#TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from load import load_data

train_x, train_y, test_x, test_y = load_data()
train_x = np.array(train_x); train_y = np.array(train_y)
test_x = np.array(test_x); test_y = np.array(test_y)
class_names = ['coding', 'non_coding']

# Shuffle the training and test set
r1 = np.arange(len(train_x))
r2 = np.arange(len(test_x))
np.random.shuffle(r1); np.random.shuffle(r2)

train_x = train_x[r1]; train_y = train_y[r1]
test_x = test_x[r2]; test_y = test_y[r2]

print("Training set: {}".format(np.array(train_y).shape))  
print("Testing set:  {}".format(np.array(test_y).shape))

column_names = ['P_val', 'Hs_val', 'F3_val']
df = pd.DataFrame(train_x, columns=column_names)
print (df.head())

#Function to build my neural network model
def build_model():
	model = keras.models.Sequential([keras.layers.Dense(5, kernel_regularizer=keras.regularizers.l2(0.001),\
		activation=tf.nn.relu,input_dim=3),\
		keras.layers.Dense(5,kernel_regularizer=keras.regularizers.l2(0.001),activation=tf.nn.relu),\
		keras.layers.Dense(5,kernel_regularizer=keras.regularizers.l2(0.001),activation=tf.nn.relu),\
		keras.layers.Dense(1, activation=tf.nn.relu)])
	optimizer = tf.train.RMSPropOptimizer(0.001)
	model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
	return model

# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 500

#Build network
model = build_model()
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
history = model.fit(train_x, train_y, epochs=EPOCHS, verbose=0, \
	validation_data=(test_x, test_y), callbacks=[early_stop, PrintDot()])

#Visualize the model's training progress using the stats stored in the history object.
def plot_history(history):
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error')
  plt.plot(history.epoch, np.array(history.history['loss']),
           label='Train Loss')
  plt.plot(history.epoch, np.array(history.history['val_loss']),
           label = 'Test loss')
  plt.legend()
  plt.show()

plot_history(history)

#Model Evaluation
score1 = model.evaluate(train_x, train_y)
print("Training accuracy: %.2f%%" % (score1[1]*100))
print('Training loss: %.3f' %(score1[0]))

print('\n')
score2 = model.evaluate(test_x, test_y)
print("Test accuracy: %.2f%%" % (score2[1]*100))
print('Test loss: %.3f' %(score2[0]))

"""
#Measure accuracy
pred = model.predict(test_x)
prediction = [1 if x>=0.5 else 0 for x in pred]
y_compare = [i[0] for i in test_y]
c = [1 if prediction[i]==y_compare[i] else 0 for i in range(len(prediction))]
accuracy = c.count(1)/len(c)
print(accuracy)
"""

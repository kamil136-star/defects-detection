# ## 1. Importing Relevant Libraries
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt 
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.datasets import load_files
from keras.utils import np_utils
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.layers import MaxPooling2D, Dropout
from keras.layers import Input, Dense, Activation,Flatten, Conv2D



# ## 2. Importing the NEU Metal Surface Defect Dataset

train_dir = './NEU Metal Surface Defects Data/train'
val_dir = './NEU Metal Surface Defects Data/valid'
test_dir='./NEU Metal Surface Defects Data/test'

# ## 3. Data Pre-processing

# Rescaling all Images by 1./255
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

# Training images are put in batches of 10
train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(200, 200),
        batch_size=10,
        class_mode='categorical')

# Validation images are put in batches of 10

validation_generator = test_datagen.flow_from_directory(
        val_dir,
        target_size=(200, 200),
        batch_size=10,
        class_mode='categorical')

# #### Setting upper Limit of Max 98% training accuracy

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy') > 0.98 ):
            print("\nReached 98% accuracy so cancelling training!")
            self.model.stop_training = True 


# ## 4. Defining the CNN Architecture

model = tf.keras.models.Sequential([
    Conv2D(32, (2,2), activation='relu', input_shape=(200, 200, 3)),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    Conv2D(64, (2,2), activation='relu'),
    MaxPooling2D(2,2),
    Dropout(0.25),

    Conv2D(128, (2,2), activation='relu'),
    MaxPooling2D(2,2),
    Dropout(0.25),

    Flatten(),
    Dense(256, activation='relu'),

    Dropout(0.25),
    Dense(6, activation='softmax')
])


model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


tf.keras.utils.plot_model(
    model,
    to_file='cnn_architecture.png',
    show_shapes=True)


# ## 5. Training the Defined CNN Model

callbacks = myCallback()
history = model.fit(train_generator,
        batch_size = 32,
        epochs=20,
        validation_data=validation_generator,
        callbacks=[callbacks],
        verbose=1, shuffle=True)



model.save('defected_steel.h5')
print (model.summary())
print(history.history)

loss_train = history.history["loss"]
np_loss_train = np.array(loss_train)

loss_val = history.history['val_loss']
np_loss_val = np.array(loss_train)

epochs = range(1, 21)
plt.plot(epochs, np_loss_train, 'g', label='Training Loss')
plt.plot(epochs, np_loss_val, 'b', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


loss_train = history.history["accuracy"]
np_loss_train = np.array(loss_train)
s = np_loss_train.shape
loss_val = history.history['val_accuracy']
np_loss_val = np.array(loss_train).reshape(s)

epochs = range(1, 21)
plt.plot(epochs, loss_train, 'g', label='Training Accuracy')
plt.plot(epochs, loss_val, 'b', label='Validation Accuracy')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

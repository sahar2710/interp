import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm

dataset_path = 'C:/Users/sahar/repos/interp/archive/asl_alphabet_train_mini'
metadata_file = pd.read_csv('C:/Users/sahar/repos/interp/archive/metadata_mini.csv')

def kernal_data(file):
    img = Image.open(file)
    kernal_data = np.array(img)
    return kernal_data

#print(kernal_data('C:/Users/sahar/repos/interp/archive/asl_alphabet_train/asl_alphabet_train/A/A1.jpg').shape)

extracted_kernal_data = []
for index_num, row in tqdm(metadata_file.iterrows()):
    file_name = os.path.join(os.path.abspath(dataset_path), str(row['folder'])+'/', str(row['file_name']))
    final_class_labels = row['class']
    data = kernal_data(file_name)
    extracted_kernal_data.append([data, final_class_labels])

extracted_kernal_data_df = pd.DataFrame(extracted_kernal_data, columns=['kernal', 'class'])
print(extracted_kernal_data_df.tail(200))

X = np.array(extracted_kernal_data_df['kernal'].tolist())
Y = np.array(extracted_kernal_data_df['class'].tolist())
y = np.array(pd.get_dummies(Y))
A_train, A_test, B_train, B_test = train_test_split(X, y, test_size=0.3, random_state=0)
num_labels = y.shape[1]
print(X.shape, y.shape, Y.shape)
print(A_train.shape, A_test.shape, B_train.shape, B_test.shape)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(2, input_shape = (20211,200,200,3)))
model.add(tf.keras.layers.Activation('sigmoid'))
model.add(tf.keras.layers.Dropout(0.2))

model.summary()
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

num_epochs = 1 
num_batch_size = 1000
checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath='saved_models/asl_interp.hdf5', verbose=1, save_best_only=True)
start = datetime.now()
model.fit(A_train, B_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(A_test, B_test), callbacks=checkpointer)
duration = datetime.now()-start
print("Training completed in time: ", duration)
test_accuracy = model.evaluate(A_test, B_test, verbose=0)
print(test_accuracy)

import pandas as pd

dataset = pd.read_csv('cancer.csv')

x = dataset.drop(columns=["diagnosis(1=m, 0=b)"])    #tolgo la collonna con questa label


y = dataset["diagnosis(1=m, 0=b)"]   #numero la colonna con questa label?


# Split the data into a training set and a testing set.

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)  #esempi, etichetta, percentuale di esempi da usare 0.2 = 20%

#output 4 insiemi, insieme addestramento, inseme dei test, etichetta add, etichetta test


#Build and train the model.


import tensorflow as tf

model = tf.keras.models.Sequential()



model.add(tf.keras.layers.Dense(256, input_shape=x_train.shape, activation='sigmoid'))
model.add(tf.keras.layers.Dense(256, activation='sigmoid'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


model.fit(x_train, y_train, epochs=1000)


model.evaluate(x_test, y_test)





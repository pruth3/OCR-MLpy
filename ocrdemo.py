
from keras.datasets import mnist
from keras.ms import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils

def bm():

	m = Sequential()
	m.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation='relu'))
	m.add(MaxPooling2D())
	m.add(Dropout(0.2))
	m.add(Flatten())
	m.add(Dense(128, activation='relu'))
	m.add(Dense(num_classes, activation='softmax'))

	m.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return m

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32')
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype('float32')

X_train = X_train / 255
X_test = X_test / 255

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]



m = bm()

m.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200)

scores = m.evaluate(X_test, y_test, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))
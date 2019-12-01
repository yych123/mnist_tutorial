import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

BATCH_SIZE = 128
NUM_CLASSES = 10
NUM_EPOCHS = 10


# download and load the data (split them between train and test sets)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# expand the channel dimension
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

# make the value of pixels from [0, 255] to [0, 1] for further process
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# convert class vectors to binary class matrics
y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)


# define the model
model = Sequential()
model.add(Conv2D(32,kernel_size=(3,3),activation="relu",input_shape=(28,28,1)))
model.add(Conv2D(16,kernel_size=(3,3),activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(32,activation="relu"))
model.add(Dense(10,activation='softmax'))


# define the object function, optimizer and metrics
model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])


# train
model.fit(x_train,y_train)

# evaluate
score_train = model.evaluate(x_train,y_train)
print('Training loss: %.4f, Training accuracy: %.2f%%' % (score_train[0],score_train[1]*100) )
score_test = model.evaluate(x_test,y_test)
print('Testing loss: %.4f, Testing accuracy: %.2f%%' % (score_test[0],score_test[1]*100) )
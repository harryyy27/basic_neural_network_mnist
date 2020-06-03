from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical
(train_images, train_labels), (test_images,test_labels) =mnist.load_data()
"""load data from keras"""
print(train_images.shape)
#60000, 28 * 28
print(len(train_labels))
#60000
print(test_images.shape)
#10000, 28*28
print(len(test_labels))
#10000
"""create neural network"""

"""instantiate sequence class - used when each layer has exactly one input and one output tensor"""
network = models.Sequential()
"""layer sizes formed from trial and error - hyperparameter tuning  - activation functions must be repeatedly differentiable to allow for complex inputs - this also allows for stacking of multiple layers """
network.add(layers.Dense(512,activation='relu',input_shape=(28 * 28,)))
""" dropping size to 10 with softmax activation to classify into various classes"""
network.add(layers.Dense(10,activation='softmax'))
""" compile network, standing for root mean square propagation - rmsprop optimizer is a form of gradient descent used for to get to answer. categorical cross entropy is the loss function used for classifying into multiple classes. this is a multinomial log loss function """
network.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])

""" this reshapes your input into the required tensor """
train_images=train_images.reshape((60000, 28 * 28))
""" normalize input by dividing by 255 (all numbers between 0 and 1) """
train_images=train_images.astype('float') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') /255
"""turning a column categorical labels from 0-9 into a matrix of binary labels """
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
""" fits network to data and labels. goes through all data 5x updating weights of each layer through a process that calculates residuals and updates weights through backwards propogation and gradient descent. generaelly a learning rate is used to ensure that incremental steps are taken in improving the model created. """
network.fit(train_images,train_labels, epochs=5, batch_size=128)
"""used to determine how well your model can classify fresh data"""
test_loss, test_acc = network.evaluate(test_images, test_labels)
print(test_loss)
#0.071
print(test_acc)
#97.86%
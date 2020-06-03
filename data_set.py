from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
(train_images, train_labels), (test_images,test_labels) =mnist.load_data()

print(train_images.shape)
print(len(train_labels))
print(train_images.ndim)
print(train_images.dtype)

digit = train_images[4]

# plt.imshow(digit, cmap=plt.cm.binary)
# plt.show()

my_slice = train_images[10:100]
two_slice = train_images[10:100,0:28,4:28]
print(my_slice)
print(my_slice.shape)
print(two_slice.shape)
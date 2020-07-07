import numpy as np
import mnist
import keras
from keras.models import Sequential #sequential: plain stack of layers
from keras.layers import Dense #dense: fully connected (dense) network layer
from keras.utils import to_categorical
#import the training image/label data
train_images = mnist.train_images()
train_labels = mnist.train_labels()

#import the testing image/label data
test_images = mnist.test_images()
test_labels = mnist.test_labels()

"""
Normalize the data
Max pixel value: 255
Divide each pixel by 255, then subtract 0.5
Final value range: -0.5 to +0.5
"""
train_images = (train_images / 255) - 0.5
test_images = (test_images / 255) - 0.5

print(train_images.shape) #60k training examples, each of which is a 28x28 pixel image
print(train_labels.shape) #60k labels

# Flatten the images.
train_images = train_images.reshape((-1, 784))
test_images = test_images.reshape((-1, 784))

"""
constructing the Sequential model:
Dense(num_nodes, activation = 'function')

softmax:
it's like a hard maximum (ex: [2, 4, 2, 1] -> [0, 1, 0, 0]) but continuous and differentiable
turns arbitary real values -> probabilities
allow us to answer with a X% level of confidence, rather than yes/no
"""
model = Sequential([
    Dense(64, activation = 'relu', input_shape = (784,)),
    Dense(64, activation = 'relu'),
    Dense(10, activation = 'softmax')
    ])

model.compile(
    optimizer='adam',#adam's pretty good at his job IG
    loss='categorical_crossentropy', #ylog(h(x)) + (1-y)log(1-h(x)). good for one-hot encoding (ex: [0 0 1 0])
    metrics=['accuracy'] #what percent of examples the model does correctly
    )

model.fit(
  train_images, # training data
  to_categorical(train_labels), # training targets
  epochs=5, #1 epoch is when the WHOLE dataset is passed through NN ONCE
  batch_size=32, #can't pass whole dataset in at once. split into batches
)

model.evaluate(
    test_images,
    to_categorical(test_labels)
    )

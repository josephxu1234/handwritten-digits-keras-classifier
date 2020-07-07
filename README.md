# mnist_digit_classifcation
Use keras to recoginze digits based on the mnist database

How it works:
- 1 import training/testing data
- 2 normalize data to range from -0.5 to +0.5
- 3 flatten the images to properly input
- 4 create Sequential model. I chose relu for hidden layers, softmax for final layer
- 5 compile the model with optimizer, loss, and metric
- 6 train the model with training images, labels, #epochs, and batch sizes
- 7 evaluate the model using training labels

# Python_Neural_Network

A simple configurable feedfroward neural network that impliments the backpropergation algorithm.

An example classifcation problem is included with training data generated with the sklearn package.

The network is initilised with the Network class, this takes 3 arguments Network(Number of features, A network topology list describing th enumber of inputs, hidden layers and outputs E.G [2.6.1], Alpha, Bias, Regularization Lambda).

NN  = Network(2,[2,6,1], 0.01, 1, 0.01).

Training is started with the NN.trainNN(Training Features, Training Labels, Epochs) function.

NN.trainNN(X_train,y_train,1000)

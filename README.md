# Simple-Neural-Network
simple NN in python

This code requires the following:
  -Python 2 (it does not work for Python 3)
  -numPy (both versions 1.8.2 and 1.9.2 have been tested; others will probably work as well)

to run the sample code, execute the following from the command line:

    python sample_run.py

to run the experiment, execute the following from the command line:
    
    python experiment.py

The following files are inlcuded:
  
  -NN.py: Contains the neural network implementation
  -NN_functions.py: Contains important functions for the Neural Network
  -perceptron.py: Contains the perceptron implementation
  -data_loader.py: Contains the code to load the data
  -sample_run.py: a sample script demonstrating usage of the code
  -data/: Directory containing the data for experimentation

Added files:
  -tuning.py: Contains the function for five-fold cross validation tuning process
  -experiment.py: Experiment code

Note:
  -I have changed the signature of function NN.create_NN() to pass the number of iterations as a parameter.


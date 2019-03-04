# Linear Regression

Step1: 

Add a class named Layer to your project. (This is not the simplest possible design for implementing linear regression. It is designed to prepare you for future assignments. So for now, just go with it.) Give this class a member variables named "activation" that can hold a vector of double-precision floating point values. Also, add an abstract method named "activate" that takes two vectors as paramets, as shown in the examples below.

Step2:

Add a class named LayerLinear that inherits from (a.k.a. extends) your Layer class. Implement the "activate" method to compute the linear equation activation=Mx+b, where x is a vector of size "inputs", and "activation" is a vector of size "outputs". M is a matrix with "outputs" rows and "inputs" columns. b is a vector of size "outputs". The vector "weights" is a big vector containing all of the values needed to fill both M and b. That is, the number of elements in "weights" will be ("outputs" + ("outputs" * "inputs")). 

Step3:

Unit testing is an important way to make sure that a portion of code works as expected. No feature is complete until it has been tested. Here is a simple unit test that you can use to make sure your code works

Step4: 

Add a method named "ordinary_least_squares" to your LayerLinear class that computes "weights". This method should accept 3 parameters: a matrix named "X", a matrix named "Y", and an uninitialized vector named "weights" to which the results will be written.

Step5:

Add a unit test for your "ordinary_least_squares" method. It should work like this:
Generate some random weights.
Generate a random feature matrix, X.
Use your LinearLayer.activate to compute a corresponding label matrix, Y.
Add a little random noise to Y.
Withhold the weights. (That is, don't use them in the next step.)
Call your ordinary_least_squares method to generate new weights.
Test whether the computed weights are close enough to the original weights. If they differ by too much, throw an exception.

Step6:

Make a new class named "NeuralNet", which holds a collection of "Layer" objects. (For this assignment, it will only hold one layer, but we will add more in future assignments.) Also add a member variable of type Vec to hold the weights of the layers. Make the NeuralNet class inherit from the SupervisedLearner class. Add a method to your NeuralNet class named "predict" that takes a vector of inputs as its parameter, and calls "Layer.activate". Add a method named "train" that uses Ordinary Least Squares to train the weights.

Step7:

Add a method to the SupervisedLearner class that computes sum-squared-error. Note that the SupervisedLearner class already contains a method that counts misclassification. Your method for computing sum-squared-error will be very similar.

Step8:

Add a method to the SupervisedLearner class that performs m-repetitions of n-fold cross-validation. (See Section 2.1.7.2.) Note that the Matrix.copyBlock method may be helpful.

Step9:

Download these features and labels. The labels give the median values of homes in Boston. The features give various statistics about the homes. (See the comments at the top of the file for more details, if you are curious.) You can use the Matrix.loadArff method to load these into data structures. Write code to perform 5 repetitions of 10-fold cross-validation, then print the root-mean-squared-error to the console.




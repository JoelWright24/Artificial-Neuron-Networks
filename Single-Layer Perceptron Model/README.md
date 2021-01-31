# Single Layer Perceptron

This Single Layer Perceptron Model, herein SLP, is a simple programme that takes in a list of training data and then can be used to predict a classification.

The training data used for this SLP is given by the "weight-height.csv" file contained in this subdirectory. The training data contains the gender of a person, given their weight and height. There are 10000 entries in this csv file. This file was provided by <https://www.kaggle.com/mustafaali96/weight-height>.

## Structure of the Python File

Object Orinted Programming, herein OOP, has been largly used to create this SLP.

The architecture of an SLP is relativly simple. An input signal comes in and is multiplied by weights and a bias is added at a summing junction. The output of the summing junction goes through an activation function to give a final output.

In the file, training takes place to update the weights and improve the results.

### The Delta Rule

The Delta rule is given by:
$$ w_{n+1} = w_{n} + \eta \left( D_{n} - I_{n} \right) $$

where:

- $w_{n}$ is the current weight
- $w_{n+1}$ is the next weight
- $\eta$ is a given learning rate
- $I_{n}$ is the input into the delta rule (the output of the activation function)
- $D_{n}$ is the desired input for a correct result.

In this code, the ouput of the acitvation function is always either a -1 or +1 and when importing data from the csv file, gender data is given in terms of -1 or +1 as well. Therefore, when the desired input matches the input into the Delta Rule, there is no change in the weights. However when there is a mis match between the input and desired value, there will be a change in the weights of $\pm 2 \eta$.

## Working Around the Training Data and Appling he Delta Rule in the Code

When updating the weights, a function is used to sample the code. It takes a given number of data and tests it against the previous weight. This way the user knows that the better weight is being chosen. The accuracy of the data is measured by running comparing the weights results to the whole of the 10000 data points.

Weakly performing weights will give a percentage of 50.0%. Through multiple passes and improvements, this percentage can be increase.

However, when optimising the weights a problem has been noticed that the algorthim used often 'gets stuck' at local peaks. This is yet to be fully resolved.

## Running This Script

When the script is run, it prints three vaules and of accuracy and their accociated weights, normally showing how the scupt optimises for accuracy against the training dataset.

It also takes in height and weights of individuales and prints out their genders based on the training. And finally the script prints a graph of a sellection of the data and shows the training line. Benieth the training line the reults will come back as female and above the training line the results will come back as male.

<div align="center">

# All About **Linear & Logistic Regression**
###### Will Chen | SHSID Data Science Group

## Key Questions

<div align="left">

Linear and logistic regression are key techniques used to understand and apply in supervised learning. By the end of the lesson, you will be able to answer the following **key questions**: 
1. What are linear and logistic regression, and what kind of problems do they solve? 
2. What's the difference between linear and logistic regression? 
3. In the age of deep learning, why do we still rely on these foundational techniques? 

<div align="center">
  
## Key Terms
<div align="left">

In order to understand this lesson, you would have a grasp of the following key concepts and terms: 

- **Features**: The input variables used to make a prediction. It goes through the neural network. When the prediction is `f(x)`, the feature would be `x` (input). 
- **Target**: The variable we are trying to predict. It is the result of a feature going through the neural network. Using the same example as above, the target is `f(x)`. 
- **Regression**: A technique used in supervised learning where the goal of the model is to predict a *continuous numerical value*. These can be predictions such as tomorrow's temperature. It's not usually used anywhere else. 
- **Classification**: A technique used in supervised learning where the goal of the model is to predict a *discrete category*. Instead of an exact number, it classifies the features into defined targets. Specifically, it outputs a *probability list of how well an input fits into each category*. These can be predictions such as marking an image to more likely to be a cat or dog. 
- **Weights**: aka Coefficients. It's one piece of data stored in a neuron that determines the feature's influence to the prediction. The influence is usually multiplication, so multiplying by a small number may mean a smaller influence, and vice versa. 
- **Biases**: aka Intercepts. Also stored in a neuron, it's a constant data that changes the baseline representiation. Weights change something based on what's given; biases change a constant amount regardless. 
- **Loss**: aka Cost. It's the difference between an erroneous output and the expected output, useful in training models. The goal of training is to minimize this function via changing the weights and biases of each neuron in each layer. 
- **Gradient descent**: An optimization that allows us to find out by what magnitude do we need to change the information within our neurons to make the loss minimized. Through iteration, it adjusts the parameters in the opposite direction of the gradient. 
- **Sigmoid**: A special function that compresses all input values into a range between 0 and 1. The key differentiator between linear and logistic. 

<div align="center">
  
## Introduction to predictive modeling
<div align="left">

### What are Linear and Logistic Regression?

**Linear Regression** and **Logistic Regression** are two of the most fundamental and universal algorithms in ML. They are both supervised learning methods, but they are used to solve different kinds of problems. 

- **Linear regression** is used for **regression** tasks, or to predict a continuous value.
  - Example: Based on past year's data, predict tomorrow's forecast. 
  - Key concept: Drawing the best-fit line for a plot of points. 

- **Logistic regression** is used for **classification** tasks. Don't let the name fool you, it's not for regression even though it's named regression, that's just the technique not the application. 
  - Example: Predict whether if a given image is a cat or a dog. 
  - Key concept: Drawing a separation line that defines the boundary between different plots of points. 

While they solve different issues, they share a similar underlying foundation in the math. Understanding linear regression is the basis to understanding logistic regression. 

### Part 1: Linear regression

As humans, we can draw a best-fit line pretty easily. Just look at the set of points on the graph and you'll have a rough estimate on which line fits best. This is because our brains are kind of built for fuzzy pattern matching stuff like this. However, it might not be mathematically the most accurate way to draw a best fit line, nor would we like for us to draw the lines ourselves, regardless of method. 

So how do we teach this to a computer, and make them do it accurately? 

Well, you might remember this slope-intercept form from math:

$$ f(x) = mx + b $$

In machine learning, this is also a core concept and you can see it in a lot of models: 
- f(x) is the target output. 
- x is the feature input. 
- m is the weight. 
- b is the bias. 

This slope-intercept form represents a line. If you want to shape the line in such a way that it fits a specific group of points, you would want to adjust the values m and b. This is exactly the same values that models adjust during training. They adjust the weights and biases for each neuron until we get a line that closely fits the "expected" point group. 

### Part 2: Logistic regression

What if our problem isn't predicting a price, but predicting a "yes" or "no" answer? The target is now a category (Cat=1, Dog=0), not a continuous number.

#### The sigmoid function

A straight line doesn't really fit our needs. If all we want is the model to tell us what it thinks the picture is, we really just need it to give us a number, between 0 and 1. For example, closer to 0 means dog, and closer to 1 means cat. 

So, we use a trick called the sigmoid function. It takes the output of a linear equation (`mx + b`) and feeds it into the sigmoid equation. 

The sigmoid function has an "S" shape. No matter what number you put into it (from negative infinity to positive infinity), it will always output a value **between 0 and 1**.

So, the logistic regression model looks like this:

$$ Probability(Animal) = Sigmoid(mx + b) $$

This output can be interpreted as the probability of the positive class. For instance, if the model outputs 0.8, it is 80% confident that the animal is a cat.

#### Decision boundaries

For our model to give us a decision instead of a number, we have to setup a decision boundary. **It's the line that sets up the distinction between results**. For a binary decision like cat or dog, it's much more simple, and the boundary is placed commonly at 0.5. If it's less than 0.5, we are more sure it is a dog than cat, and vice versa. 

#### Binary cross entropy

Because our predictions are now probabilities, the traditional loss function for most models, Mean-Squared Error, is no longer the best loss function. Instead, logistic regression uses a loss function called **binary cross-entropy** (or los loss). It heavily "punishes" the model when it makes a confident but incorrect prediction. For example, if the model predicts a 99% chance of a cat who is actually a dog, the loss will be very high and the model will be severely "punished". 

Apart from this, the rest can be basically the same. They both use concepts such as chain rule and gradient descent to calculate the specific neurons, magnitude, and direction of change. 

<div align="center">

### Conclusion
<div align="left">

Linear and Logistic Regression are the foundational pillars of predictive modeling. They demonstrate the core process of machine learning: defining a model, measuring its error with a **loss function**, and iteratively improving it using an optimizer like **gradient descent**.

- **Linear Regression** fits a line to data to predict **continuous values**.
- **Logistic Regression** adapts this line with a **sigmoid function** to predict a probability for **classification tasks**.

Although simpler than deep neural networks and more complex topics, their importance is still monumental. They are fast, interpretable, and serve as the starting point for beginners. In fact, a single neuron in a neural network performing a classification task is essentially a logistic regression unit. 

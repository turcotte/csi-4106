# Introduction to Artificial Neural Networks

CSI4106 Introduction to Artificial Intelligence

> **IMPORTANT:**
>
> **Deadline**: Assignment 2 must be submitted no later than October 26, 2026, at 11 PM. Please refer to the assignment description available on Brightspace.

## Prepare

- Russell and Norvig ([2020](#ref-Russell:2020aa)), pages 750–788

### Watch 3Blue1Brown videos on neural networks

- [**But what is a Neural Network?**](https://youtu.be/aircAruvnKk) (19 minutes)
- [**Gradient descent, how neural networks learn**](https://youtu.be/IHZwWFHWa-w) (21 minutes)
- [**What is backpropagation really doing?**](https://youtu.be/Ilg3gGewQ5U) (14 minutes)
- [**Backpropagation calculus**](https://youtu.be/tIeHLnjs5U8) (10 minutes)

### Narrative of PyTorch’s inception

- [PyTorch (2024) Official PyTorch Documentary - Powering the AI Revolution](https://www.youtube.com/watch?v=rgP_LBtaUEc) (36 minutes)

## Participate

- [slides](slides.llms.md) ([PDF](slides.pdf), [Jupyter Notebook](slides.ipynb))

## Practice

- [Circular Separability](CircularSeparability): To complement your exploration of TensorFlow Playground (below), I have developed a notebook focusing on feature engineering.

- [TensorFlow Playground](https://playground.tensorflow.org/)

  - **Dataset Options**: Users can choose from four types of datasets: circular, XOR, Gaussian, and spiral.
  - **Feature Engineering**: Enables the creation of new features to improve model performance.
  - **Model Architecture**: Allows customization of neural network architecture, including varying the number of layers and neurons per layer.
  - **Hyperparameter Tuning**: Provides options to adjust learning rate, activation functions, regularization techniques, and task specifications to observe their effects on model training.
  - **Suggestion 1**: For the Gaussian dataset, which is linearly separable, configure a network without hidden layers and a single output neuron using the sigmoid activation function. This setup effectively constructs a logistic regression model.
  - **Suggestion 2**: The circular dataset is not linearly separable using only the original features \\x_1\\ and \\x_2\\. However, by creating new features, \\x_1^2\\ and \\x_2^2\\, the problem becomes linearly separable in the transformed feature space. A network with no hidden layers and a single output node is sufficient for this task.

## References

Russell, Stuart, and Peter Norvig. 2020. *Artificial Intelligence: A Modern Approach*. 4th ed. Pearson. <http://aima.cs.berkeley.edu/>.

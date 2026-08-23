# Logistic regression

CSI4106 Introduction to Artificial Intelligence

> **IMPORTANT:**
>
> **Deadline**: Assignment 1 must be submitted no later than October 5, 2026, at 11 PM. Please refer to the assignment description available on Brightspace.

## Prepare

- Logistic regression: Russell and Norvig ([2020](#ref-Russell:2020aa)), pages 684-686
- Watch [Machine Learning and Logistic Regression](https://youtu.be/AX-ZEC-71DI), IBM Technology.

## Participate

- [slides](slides.llms.md) ([PDF](slides.pdf), [Jupyter Notebook](slides.ipynb))

## Practice

In class, we developed a logistic regression model for handwritten digit recognition using a dataset from UCI ML. This dataset comprises 1797 images of size \\8 \times 8\\. The [MNIST](https://www.openml.org/search?type=data&sort=runs&id=554&status=active) (`mnist_784`) dataset contains 70,000 images of size \\28 \times 28\\. The following [example](https://scikit-learn.org/dev/auto_examples/linear_model/plot_sparse_logistic_regression_mnist.html), from the `sklearn` website, uses this dataset and graphically presents the coefficients (\\\theta\\) for each of the 10 models. You can load this model as follows:

``` python
from sklearn.datasets import fetch_openml  

X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
```

## References

Russell, Stuart, and Peter Norvig. 2020. *Artificial Intelligence: A Modern Approach*. 4th ed. Pearson. <http://aima.cs.berkeley.edu/>.

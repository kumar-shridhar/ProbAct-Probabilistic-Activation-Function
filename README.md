# ProbAct-Probabilistic-Activation-Function

Official PyTorch implementation of the paper : ![ProbAct: A Probabilistic Activation Function for Deep Neural Networks](https://arxiv.org/abs/1905.10761).
![ProbAct](Visualization/ProbAct.png)

## Why ProbAct

Most of the activation functions currently used are deterministic in nature, whose input-output relationship is fixed. In this work, we propose a probabilistic activation function, called *ProbAct*. The output value of ProbAct is sampled from a normal distribution, with the mean value same as the output of ReLU and with a fixed or trainable variance for each element. In the trainable ProbAct, the variance of the activation distribution is trained through back-propagation. We also show that the stochastic perturbation through ProbAct is a viable generalization technique that can prevent overfitting.

## Accuracy Comparison

![Comparison with other activation Functions](Visualization/ComparisonResultsProbAct.png)

## Overfitting Comparison

![Test-Train Comparison on CIFAR100](Visualization/OverfittingCIFAR100.png)


Cite the authors if you find the work useful:

```
@article{lee2019probact,
  title={ProbAct: A Probabilistic Activation Function for Deep Neural Networks},
  author={Lee, Joonho and Shridhar, Kumar and Hayashi, Hideaki and Iwana, Brian Kenji and Kang, Seokjun and Uchida, Seiichi},
  journal={arXiv preprint arXiv:1905.10761},
  year={2019}
}

```











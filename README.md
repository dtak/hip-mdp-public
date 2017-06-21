# Robust and Efficient Transfer Learning with Hidden Parameter Markov Decision Processes

This repository contains an implementation of a framework for training and testing Hidden Parameter Markov Decision Processes [(on arXiv.org)](https://arxiv.org/abs/1706.06544) and other RL benchmarks.

### Abstract

We introduce a new formulation of the Hidden Parameter Markov Decision
Process (HiP-MDP), a framework for modeling families of related tasks
using low-dimensional latent embeddings. We replace the original
Gaussian Process-based model with a Bayesian Neural Network. Our new
framework correctly models the joint uncertainty in the latent weights
and the state space and has more scalable inference, thus expanding
the scope the HiP-MDP to applications with higher dimensions and more
complex dynamics.

## Example

The juypter notebook titled toy_example.ipynb contains an example of training and testing a HiP-MDP with embedded weights on a 2-D navigation domain.

### Prerequisites

```
Python 2.7.12
tensorflow 0.12.1
numpy 1.11.1
autograd 1.1.7
seaborn 0.7.1
```

## Authors

* **Taylor Killian**
* **Samuel Daulton**
* **Finale Doshi-Velez**
* **George Konidaris**

## License

The source code and documentation are licensed under the terms of the [MIT License](https://opensource.org/licenses/MIT).

## Acknowledgments
* Harvard DTAK
* Harvard Paulson School of Engineering and Applied Sciences
* MIT Lincoln Laboratory
* Alpha divergence bayesian neural network adapted from Jose Miguel Hernandez Lobato's original code
* HIV treatment and acrobot simulators adapted from RLPy's [simulators](https://bitbucket.org/rlpy/rlpy/src/master/rlpy/Domains)
* Priority queue adapted from Kai Arulkumaran's [Atari](https://github.com/Kaixhin/Atari) repository
* Deep Q Network follows [Juliani's implementation](https://github.com/awjuliani/DeepRL-Agents)



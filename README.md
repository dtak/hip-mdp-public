# Robust and Efficient Transfer Learning with Hidden Parameter Markov Decision Processes

This repository contains an implementation of a framework for training and testing Hidden Parameter Markov Decision Processes (to appear at NIPS 2017, pre-print [available here](https://arxiv.org/abs/1706.06544)) and other RL benchmarks.

### Abstract

We introduce a new formulation of the Hidden Parameter Markov Decision
Process (HiP-MDP), a framework for modeling families of related tasks
using low-dimensional latent embeddings. We replace the original
Gaussian Process-based model with a Bayesian Neural Network. Our new
framework correctly models the joint uncertainty in the latent weights
and the state space and has more scalable inference, thus expanding
the scope the HiP-MDP to applications with higher dimensions and more
complex dynamics.

## Repository Contents

- `Example` We demonstrate the training and testing of a HiP-MDP with embedded weights on the 2-D navigation domain (Grid) in [toy_example.ipynb](./toy_example.ipynb).
- `Simulators:` source code to run various control domains. In particular,
  - [Acrobot](./acrobot_simulator/) The Acrobot, an inverted double pendulum, introduced by Richard Sutton (1996) and summarized in [Reinforcement Learning: An Introduction](http://incompleteideas.net/sutton/book/the-book-2nd.html).
  - [Grid](./grid_simulator/) A toy 2-D navigation domain developed to illustrate the concept of transfer through a HiP-MDP.
  - [HIV](./hiv_simulator/) Determining effective treatment protocols for simulated patients with HIV tracking their physiological response to separate classes of treatments. First introduced as an RL domain by Damien Ernst, et al. (2006).
- `Utilities:` scripts used to build, combine and run the components of the HiP-MDP.
  - [BNNs](./BayesianNeuralNetwork.py) contains the class module and support functions to build, train and evaluate Bayesian Neural Networks with $\alpha$-divergence minimization.
  - [Experience Replay](./ExperienceReplay.py) contains the class to build and sample experience buffers, used for training neural networks with experience replay. Can be used to sample uniformly or in a prioritized fashion (after [Schaul, et al.](https://arxiv.org/abs/1511.05952) (2015)).
  - [Priority Queue](./PriorityQueue.py) is a class module used to facilitate prioritized experience replay which implements a (Max) Binary Heap.
  - [DQN](./Qnetwork.py) is a class module that contains the code to build deep Q-networks. Currently defaulted to be a Double DQN ([van Hasselt, et al.](http://dl.acm.org/citation.cfm?id=3016191) (2016)).


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
* **George Konidaris**
* **Finale Doshi-Velez**


## License

The source code and documentation are licensed under the terms of the [MIT License](https://opensource.org/licenses/MIT).

## Acknowledgments
* Harvard DTAK
* Harvard Paulson School of Engineering and Applied Sciences
* MIT Lincoln Laboratory Lincoln Scholars Program
* Alpha divergence bayesian neural network adapted from Jose Miguel Hernandez Lobato's [original code](https://bitbucket.org/jmh233/code_black_box_alpha_icml_2016) 
* HIV treatment and acrobot simulators adapted from RLPy's [simulators](https://bitbucket.org/rlpy/rlpy/src/master/rlpy/Domains)
* Priority queue adapted from Kai Arulkumaran's [Atari](https://github.com/Kaixhin/Atari) repository
* Deep Q Network follows [Juliani's implementation](https://github.com/awjuliani/DeepRL-Agents)



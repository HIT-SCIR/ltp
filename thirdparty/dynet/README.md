# DyNet
The Dynamic Neural Network Toolkit

DyNet (formerly known as [cnn](http://github.com/clab/cnn-v1)) is a neural network library developed by Carnegie Mellon University and many others. It is written in C++ (with bindings in Python) and is designed to be efficient when run on either CPU or GPU, and to work well with networks that have dynamic structures that change for every training instance. For example, these kinds of networks are particularly important in natural language processing tasks, and DyNet has been used to build state-of-the-art systems for [syntactic parsing](https://github.com/clab/lstm-parser), [machine translation](https://github.com/neubig/lamtram), [morphological inflection](https://github.com/mfaruqui/morph-trans), and many other application areas.

Read the [documentation](http://dynet.readthedocs.io/en/latest/) to get started, and feel free to contact the [dynet-users group](https://groups.google.com/forum/#!forum/dynet-users) group with any questions (if you want to receive email make sure to select "all email" when you sign up). We greatly appreciate any bug reports and contributions, which can be made by filing an issue or making a pull request through the [github page](http://github.com/clab/dynet).

You can also read more technical details in our [technical report](https://arxiv.org/abs/1701.03980). If you use DyNet for research, please cite this report as follows:

    @article{dynet,
      title={DyNet: The Dynamic Neural Network Toolkit},
      author={Graham Neubig and Chris Dyer and Yoav Goldberg and Austin Matthews and Waleed Ammar and Antonios Anastasopoulos and Miguel Ballesteros and David Chiang and Daniel Clothiaux and Trevor Cohn and Kevin Duh and Manaal Faruqui and Cynthia Gan and Dan Garrette and Yangfeng Ji and Lingpeng Kong and Adhiguna Kuncoro and Gaurav Kumar and Chaitanya Malaviya and Paul Michel and Yusuke Oda and Matthew Richardson and Naomi Saphra and Swabha Swayamdipta and Pengcheng Yin},
      journal={arXiv preprint arXiv:1701.03980},
      year={2017}
    }

[![Build Status](https://travis-ci.org/clab/dynet.svg?branch=master)](https://travis-ci.org/clab/dynet)
[![Doc build Status](https://readthedocs.org/projects/dynet/badge/?version=latest)](http://dynet.readthedocs.io/en/latest/)

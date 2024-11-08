<p align="center"><img src="https://relbench.stanford.edu/img/logo.png" alt="logo" width="600px" /></p>

----

[![website](https://img.shields.io/badge/website-live-brightgreen)](https://relbench.stanford.edu)
[![PyPI version](https://badge.fury.io/py/relbench.svg)](https://badge.fury.io/py/relbench)
[![Testing Status](https://github.com/snap-stanford/relbench/actions/workflows/testing.yml/badge.svg)](https://github.com/snap-stanford/relbench/actions/workflows/testing.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Follow%20%40RelBench)](https://twitter.com/RelBench)

<!-- **Get Started:** loading data &nbsp; [<img align="center" src="https://colab.research.google.com/assets/colab-badge.svg" />](https://colab.research.google.com/drive/1PAOktBqh_3QzgAKi53F4JbQxoOuBsUBY?usp=sharing), training model &nbsp; [<img align="center" src="https://colab.research.google.com/assets/colab-badge.svg" />](https://colab.research.google.com/drive/1_z0aKcs5XndEacX1eob6csDuR4DYhGQU?usp=sharing). -->


<!-- [<img align="center" src="https://relbench.stanford.edu/img/favicon.png" width="20px" />   -->
[**Website**](https://relbench.stanford.edu) | [**Position Paper**](https://proceedings.mlr.press/v235/fey24a.html) |  [**Benchmark Paper**](https://arxiv.org/abs/2407.20060) | [**Mailing List**](https://groups.google.com/forum/#!forum/relbench/join)

# Overview

This is the Milestone Code repo for the "Positional Encodings for Relational Datasets" Project. The main python script for training and testing models is examples/rpe_gnn_node.py. New implemented models are in Relbench/models directory.

# Cite RelBench

If you use RelBench in your work, please cite our position and benchmark papers:

```bibtex
@inproceedings{rdl,
  title={Position: Relational Deep Learning - Graph Representation Learning on Relational Databases},
  author={Fey, Matthias and Hu, Weihua and Huang, Kexin and Lenssen, Jan Eric and Ranjan, Rishabh and Robinson, Joshua and Ying, Rex and You, Jiaxuan and Leskovec, Jure},
  booktitle={Forty-first International Conference on Machine Learning}
}
```

```bibtex
@misc{relbench,
      title={RelBench: A Benchmark for Deep Learning on Relational Databases},
      author={Joshua Robinson and Rishabh Ranjan and Weihua Hu and Kexin Huang and Jiaqi Han and Alejandro Dobles and Matthias Fey and Jan E. Lenssen and Yiwen Yuan and Zecheng Zhang and Xinwei He and Jure Leskovec},
      year={2024},
      eprint={2407.20060},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2407.20060},
}
```

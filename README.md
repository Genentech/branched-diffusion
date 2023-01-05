# Hierarchically branched diffusion models for efficient and interpretable multi-class conditional generation

<p align="center">
    <img src="references/thumbnail.png" width=400px />
</p>

### Introduction

Diffusion models have achieved justifiable popularity by attaining state-of-the-art performance in generating realistic objects from seemingly arbitrarily complex data distributions, including when conditioning generation on labels.

Unfortunately, however, their iterative nature renders them very computationally inefficient during the sampling process. For the multi-class conditional generation problem, we propose a novel, structurally unique framework of diffusion models which are hierarchically branched according to the inherent relationships between classes.

Branched diffusion models offer major improvements in efficiently generating samples from multiple classes. We also showcase several other advantages of branched diffusion models, including ease of extension to novel classes in a continual-learning setting, and a unique interpretability that offers insight into these generative models. Branched diffusion models represent an alternative paradigm to their traditional linear counterparts, and can have large impacts in how we use diffusion models for efficient generation, online learning, and scientific discovery. 

See the [corresponding paper](https://arxiv.org/abs/2212.10777) for more information.

This repository houses all of the code used to generate the results for the paper, including code that processes data, trains models, implements branched diffusion models, and generates all figures in the paper.

### Citing this work

If you found branched diffusion models to be helpful for your work, please cite the following:

Tseng, A.M., Biancalani, T., Shen, M., Scalia, G. Hierarchically branched diffusion models for efficient and interpretable multi-class conditional generation. arXiv (2022) [Link](https://arxiv.org/abs/2212.10777)

[\[BibTeX\]](references/bibtex.bib)

### Description of files and directories

```
├── Makefile    <- Installation of dependencies
├── data    <- Contains data for training and downstream analysis
│   ├── raw    <- Raw data, directly downloaded from the source
│   ├── interim    <- Intermediate data mid-processing
│   ├── processed    <- Processed data ready for training or analysis
│   └── README.md    <- Description of data
├── models
│   └── trained_models    <- Trained models
├── notebooks    <- Jupyter notebooks that explore data, plot results, and analyze results
│   └── figures    <- Jupyter notebooks that create figures
├── results    <- Saved results
├── README.md    <- This file
└── src    <- Source code
    ├── feature    <- Code for data loading and featurization
    ├── model    <- Code for model architectures and training
    ├── analysis    <- Code for analyzing results
    └── plot    <- Code for plotting and visualization
```

# Dynamic Nonparametric Functional Graphical Model (CACO)

This repository contains the official Python implementation of the Conditional Additive Covariance Operator (CACO) algorithm for estimating dynamic functional graphical models.

## Overview
CACO is a novel nonparametric framework designed to estimate dynamic functional graphical models in the presence of external covariates. It extends the notion of functional additive conditional independence (FACI) to dynamic settings, completely bypassing strict distributional assumptions (e.g., Gaussian or copula Gaussian). 

## Requirements
To run the codes, please make sure your Python environment has the following packages installed:
- `numpy`
- `scipy`
- `scikit-learn`
- `matplotlib`
- `tqdm`

## Example Script / Quick Start

To reproduce the 5-fold cross-validation and sensitivity analysis (comparing RBF vs. Polynomial kernels, and parameter scaling) on the Model I (Tree) topology as discussed in our paper, simply run the example script: 

cv_sensitivity.py

This script will automatically generate synthetic functional data, apply the CACO algorithm, and output the Area Under the ROC Curve (AUC) for the estimated dynamic graphs.

## Core File

scaco.py: Contains the core algorithmic implementations (e.g., coordinate mapping, kernel matrices generation, and precision operator norm calculation).

cv_sensitivity.py: An example script demonstrating how to tune hyperparameters and test different kernel functions using $K$-fold cross validation.

## Citation

If you find this code useful for your research, please cite our paper (BibTeX to be updated upon publication).

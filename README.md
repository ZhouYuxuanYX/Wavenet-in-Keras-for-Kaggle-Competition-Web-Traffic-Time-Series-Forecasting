# Sequence-to-Sequence-Wavenet-in-Keras-for-Kaggle-Competition-Web-Traffic-Time-Series-Forecasting
Sequence to Sequence Model based on Wavenet instead of LSTM implemented in Keras

# Web Traffic Forecasting
My solution for the Web Traffic Forecasting competition hosted on Kaggle.

## Competition Goal
The training dataset consists of approximately 145k time series.  Each of these time series represents a number of daily views of a different Wikipedia article, starting from July 1st, 2015 up until September 10th, 2017. The goal is to forecast the daily views between September 13th, 2017 and November 13th, 2017 for each article in the dataset.

The evaluation metric for the competition is symmetric mean absolute percentage error (SMAPE), but here we simply adopt mean absolute error(MAE) as loss function.


## Introduction to Wavenet
The model architecture is similar to WaveNet, consisting of a stack of dilated causal convolutions, as demonstrated in the [diagram](https://deepmind.com/blog/wavenet-generative-model-raw-audio/) below. For more details, see van den Oord's [paper](https://arxiv.org/abs/1609.03499).

<p align="center">
  <img src="figures/wavenet.gif">

</p>

** Causal Convolution **:


Below are some sample forecasts to demonstrate some of the patterns that the network can capture.  The forecasted values are in yellow, and the ground truth values (not used in training or validation) are shown in grey.  The y-axis is log transformed.

<img src="figures/figure_1.png" width="440"> <img src="figures/figure_2.png" width="440">
<img src="figures/figure_5.png" width="440"> <img src="figures/figure_4.png" width="440">
<img src="figures/figure_6.png" width="440"> <img src="figures/figure_3.png" width="440">


## Requirements
12 GB GPU (recommended), Python 2.7

Python packages:
  - numpy==1.13.1
  - pandas==0.19.2
  - scikit-learn==0.18.1
  - tensorflow==1.3.0

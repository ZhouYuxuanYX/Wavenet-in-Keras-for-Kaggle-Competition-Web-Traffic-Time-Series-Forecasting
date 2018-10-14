# Wavenet-in-Keras-for-Kaggle-Competition-Web-Traffic-Time-Series-Forecasting
Sequence to Sequence Model based on Wavenet instead of LSTM implemented in Keras

# Web Traffic Forecasting
To download the data and know more about the competition, see [here](https://www.kaggle.com/c/web-traffic-time-series-forecasting/kernels?sortBy=voteCount&group=everyone&pageSize=20&competitionId=6768)

## Competition Goal
The training dataset consists of approximately 145k time series.  Each of these time series represents a number of daily views of a different Wikipedia article, starting from July 1st, 2015 up until September 10th, 2017. The goal is to forecast the daily views between September 13th, 2017 and November 13th, 2017 for each article in the dataset.

The evaluation metric for the competition is symmetric mean absolute percentage error (SMAPE), but here we simply adopt mean absolute error(MAE) as loss function.


## Introduction to Wavenet
The model architecture is similar to WaveNet, consisting of a stack of dilated causal convolutions, as demonstrated in the [diagram](https://deepmind.com/blog/wavenet-generative-model-raw-audio/) below. For more details, see van den Oord's [paper](https://arxiv.org/abs/1609.03499).

<p align="center">
  <img src="figures/wavenet.gif">

</p>

**Causal Convolution**:

The figure below shows a causal structure, which guarantees that the current time step is only influenced by the previous time steps. Then an expression of the conditional probability could be established. That is to say, we assume that the current value is conditioned on the previous values in a time sequence. 


<p align="center">
  <img src="figures/WaveNet_causalconv.png">

</p>

**Dilated Convolution**:

But as can be seen, the reception field is quite small with a limited number of stacks, and it results in poor performance handling long-term dependencies. So the idea of dilated convolution is employed. In a dilated convolution layer, filters are not applied to inputs in a simple sequential manner, but instead skip a constant dilation rate inputs in between each of the inputs they process, as in the WaveNet diagram below. By increasing the dilation rate multiplicatively at each layer (e.g. 1, 2, 4, 8, …), we can achieve the exponential relationship between layer depth and receptive field size that we desire. The figure below ilustrates the effect of dilation.

<p align="center">
  <img src="figures/WaveNet_dilatedconv.png">

</p>

## Introduction to Sequence-to-Sequence Model

**RNN based seq2seq model**:

A seq2seq model is mainly used in NLP tasks such as machine translation and often based on LSTM or GRU structure. It has encoder, decoder and intermediate step as its main components, mapping an arbitrarily long input sequence to an arbitrarily long output sequence with an intermediate encoded state.:

<p align="center">
  <img src="figures/seq2seq.png">

</p>

In comparison to fully connected feed forward neural networks, recurrent neural networks has no longer the requirement a fixed-sized input and considers naturally the relation between previous and current time steps. In addition, LSTM or GRU are advanced RNN structures, which increase the ability of capturing long-term dependencies, by forcing a approximately constant back-propagation error flow during training.

However, due to the recurrent calculation for each time step, parrellelization is impossible for training theses networks. And it's a big disadvantage in the big data era. Even the input time range for a LSTM can not be arbitrary long in reality, and it is in fact severly limited by the training mechanism of RNN.

**Wavenet based approach**:

With Wavenet, the training procedure for all the time steps in the input can be parrellelized. We just let the output sequence be one time step ahead of the input sequence, and at every time step of the output, the value is only influenced by the previous steps in the input.

As for the inference stage, it yields every time only the prediction one step ahead as in the LSTM approach. But we don't need to define a distinct model for inferencing here. In each Iteration, the last point of the output sequence is selected as the prediction one step ahead of the previous iteration, and it is in turn concatenated to the input sequence, in order to predict one step further in the future. 

# About this Project

Inpired from the core ideas of Wavenet: dilated causal convolution, a simpler version of it is implemented in Keras in my Project, disregarding the residual blocks used in the original paper, which is mainly employed to make deep neural networks easier to train. And this is not problem here for my project.

And there are some crucial factors affecting the model performance:

* **Filter Size:** 



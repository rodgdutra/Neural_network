# Redes neural com python

Este repositório contém vários exemplos de  aplicações de redes neurais artificiais para a solução de 2 tipos de problemas principais, classificação e regressão em séries temporais.

## Rede Neural aplicada a partir do zero
Há um exemplo deste repositório aplicando uma Rede Neural do tipo Multiple layer perceptron (MLP) usando python para fazer isso.
* [rede neural MLP tradicional](https://github.com/rodgdutra/Neural_network/blob/master/classic_neural_net/classic_net.ipynb)

## Classificação

### Classificador de cancer de mama
Esse exemplo de classificação foi aplicado a um conjunto de dados de mamografia para detectar a gravidade de um câncer de mama.
* [Classificação do câncer de mama](https://github.com/rodgdutra/Neural_network/blob/master/mamografy_dataset_example/mamografy_classification.ipynb)

### Redes neurais autoassociativas competitivas aplicadas na classificação de tipo de vinho
Esse exemplo mostra a aplicação primeiramente de uma rede neural do tipo Multiple Layer Perceptron (MLP) em uma arquitetura tradicional e a compara com uma MLP em formato autoassociativo (ou autoencoder) competitivo para a solução do mesmo problema.
* [Classificação do vinho](https://github.com/rodgdutra/Neural_network/tree/master/wine_dataset_example)

## Previsão de um passo a frente em séries temporais

### Previsão da velocidade do vento.
Esse exemplo aplica redes neurais do tipo MLP para a previsão da velocidade do vento, tendo como entrada o comportamento passado dessa velocidade.
* [Previsão de vento](https://github.com/rodgdutra/Neural_network/blob/master/time_series_wind/wind_prediction.ipynb)


### Previsão do índice BOVA11.
Nesse exemplo é aplicado a previsão de 1 passo a frente da série histórica de valores do indice BOVA11 utilizando uma rede neural do tipo Long short term memory (LSTM) e o modelo estatístico ARIMA para comparação.
* [Previsão BOVA11](https://github.com/rodgdutra/Neural_network/blob/master/time_series_bovespa/ibov_time_series.ipynb)


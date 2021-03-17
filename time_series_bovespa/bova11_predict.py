# %% [markdown]
# ## This notebook contains a full-time series analysis of BOVA11 ETF stock
# This notebook aims to analysis and predicts a one-step prediction of BOVA11 stocks.

# %%
import pprint
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from minepy import MINE
import numpy as np  # Run PTP simulation
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import allantools
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.stats import moment
from scipy.signal import decimate, resample
from scipy.spatial.distance import correlation as dist_corr
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Dropout, Lambda
from keras.callbacks import EarlyStopping
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from statsmodels.tsa.arima_model import ARIMA

get_ipython().run_line_magic("matplotlib", "inline")
get_ipython().run_line_magic("load_ext", "autoreload")
get_ipython().run_line_magic("autoreload", "2")

# %% [markdown]
# ## Time series analysis
#   [1] Define a weak stationary time series if the mean function $ E[x(t)] $ is independent of $ t $, if the autocoraviation function $Cov (x(t+h), x(t))$  is independent of $ t $ for each $h$ and if $E[x^2[n]]$ is finite for each $n$.
#
#   To perform the weak stationary test, the mean function and the autocovariation function ware applied over rolling windows, since its sampled data. Thus, the window size has an impact over the functions interpretations, the window represents the interval in which the stationary hypothesis is tested.
#
#   As a sanity check, the week stationary test will be applied over white noise data and random-walk data.[1] Proofs mathematically that white noise is stationary, and that random walk data is non-stationary, by the weak stationary definition.
#
#   Besides this definition, the $ statsmodels $ library has the Augmented Dickey-Fuller unit root test.
#   The Augmented Dickey-Fuller test can be used to test for a unit root in a univariate process in the presence of serial correlation.
#
#   ### References
#   [1] Brockwell, Peter J., and Richard A. Davis. Introduction to time series and forecasting. springer, 2016.

# %%
def stationary_test(entry, delta=200, ad=False, std=False):
    window_size = int(len(entry) / 15)
    # Weak stationary test
    # Mean function
    mean_y = []
    mean_x = []

    s_moment_y = []
    std_y = []
    n_data = len(entry)
    for i in range(0, int(n_data - window_size)):
        # Roling window start and end
        n_start = i
        n_end = n_start + window_size
        # Mean, standard deviation and second moment calculation
        mean_y_i = np.mean(entry[n_start:n_end])
        s_moment_y_i = moment(entry[n_start:n_end], moment=2)
        std_y_i = np.std(entry[n_start:n_end])
        # Saving the results
        mean_y.append(mean_y_i)
        mean_x.append(n_end)
        s_moment_y.append(s_moment_y_i)
        std_y.append(std_y_i)

    # Autocovariance function
    acov_y = []
    acov_x = []
    n_data = len(entry)
    for i in range(0, int(n_data - window_size - delta)):
        n_start = i
        n_end = n_start + window_size
        acov_y_i = np.cov(entry[n_start:n_end], entry[n_start + delta : n_end + delta])[
            0
        ][0]
        acov_y.append(acov_y_i)
        acov_x.append(n_end)
    if ad:
        result = adfuller(entry)
        print("ADF Statistic: %f" % result[0])
        print("p-value: {0}".format(result[1]))
        print("Critical Values:")
        for key, value in result[4].items():
            print("\t%s: %.3f" % (key, value))
        # if the p-value < 0.05  and the adf statistic is less than
        # critical values the series is stationary or is time independent

    return [mean_x, mean_y], [acov_x, acov_y], s_moment_y, std_y


# %% [markdown]
# ## Weak stationary test on well-Known data
# [1] Proofs that white noise data is stationary and random walk data is non-stationary. This way, a sanity test would be to test the stationarity of these data.

# %%
# White noise data
white_noise = allantools.noise.white(1000)
random_walk = allantools.noise.brown(1000)

# To help us  visualise the constance in the cov plot lets first scale the entry
white_noise = white_noise  # /(np.amax(white_noise))
random_walk = random_walk  # /np.amax(np.abs(random_walk))
plt.figure(figsize=(8, 4), dpi=80, facecolor="w", edgecolor="k")
plt.subplot(121)
plt.plot(white_noise, "b", label="white noise")
plt.xlabel("samples index")
plt.title("White noise")
plt.subplot(122)
plt.plot(random_walk, "b", label="random_walk")
plt.xlabel("samples index")
plt.title("random_walk")
plt.show()

plt.figure(figsize=(8, 4), dpi=80, facecolor="w", edgecolor="k")
plt.subplot(121)
plt.ylabel("Probability")
plt.hist(white_noise, 50)
plt.subplot(122)
plt.hist(random_walk, 50)
plt.show()

mean_w, cov_w, s_moment_w, std_w = stationary_test(white_noise, delta=100, ad=False)
mean_r, cov_r, s_moment_r, std_r = stationary_test(random_walk, delta=100, ad=False)

plt.figure(figsize=(8, 4), dpi=80, facecolor="w", edgecolor="k")
plt.subplot(121)
plt.plot(white_noise, "b", label="white noise")
plt.plot(mean_w[0], mean_w[1], "r", label="mean function")
plt.legend()
plt.subplot(122)
plt.plot(random_walk, "b", label="random_walk")
plt.plot(mean_r[0], mean_r[1], "r", label="mean function")
plt.legend()
plt.xlabel("samples index")
plt.title("Mean function ")
plt.show()

plt.figure(figsize=(8, 4), dpi=80, facecolor="w", edgecolor="k")
plt.subplot(121)
plt.plot(white_noise, "b", label="white noise")
plt.plot(mean_w[0], s_moment_w, "r", label="Second moment function")
plt.legend()
plt.subplot(122)
plt.plot(random_walk, "b", label="random_walk")
plt.plot(mean_r[0], s_moment_r, "r", label="Second moment function")
plt.xlabel("samples index")
plt.title("Second moment function ")
plt.show()

plt.figure(figsize=(8, 4), dpi=80, facecolor="w", edgecolor="k")
plt.subplot(121)
plt.plot(white_noise, "b", label="white noise")
plt.plot(cov_w[0], cov_w[1], "g", label="Auto covariance function")
plt.legend()
plt.subplot(122)
plt.plot(random_walk, "b", label="random_walk")
plt.plot(cov_r[0], cov_r[1], "g", label="Auto covariance function")
plt.xlabel("samples index")
plt.title("ACF ")
plt.legend()
plt.show()

plt.figure(figsize=(8, 4), dpi=80, facecolor="w", edgecolor="k")
plt.subplot(121)
plt.plot(white_noise, "b", label="white noise")
plt.plot(mean_w[0], std_w, "r", label="Standard deviation function")
plt.legend()
plt.subplot(122)
plt.plot(random_walk, "b", label="random_walk")
plt.plot(mean_r[0], std_r, "r", label="Standard deviation function")
plt.xlabel("samples index")
plt.title("Standard deviation function ")
plt.show()

# White noise
plt.figure
pd.plotting.autocorrelation_plot(white_noise)
plt.show()

# random walk
plt.figure
pd.plotting.autocorrelation_plot(random_walk)
plt.show()

# %% [markdown]
# ### Sanity test analysis
# The test showed that the white noise presents a mean, covariance, standard deviation functions close to a constant in time, on the other hand, the random walk presents the same functions varying in time. Thus, by the functions presented here, the white noise is stationary, and the random walk is non-stationary, note that we are analyzing one realization of those random process, hence with the close price data we will have also only 1 realization to test if the data is either stationary or not.
#
# Besides that, the autocorrelation function on the random walk shows a slow decaying behavior and for several lags, the ACF presents values above the confidence interval. Thus, the ACF is another indication of the random walk non-stationary behavior.
# The ACF applied on the white noise function presents, on almost every lag, values bellow the confidence lines. Hence, the white noise ACF shows an indication of the white noise stationarity.
# %% [markdown]
# ## Importing time series data

# %%
# Importing Data
data_csv = "bova11.csv"
dataframe = pd.read_csv(data_csv)
data = dataframe.values
price = data[:, 1]
open_p = data[:, 2]
high = data[:, 3]
low = data[:, 4]
volume = data[:, 5]
# price = np.array(price[1100:],dtype=float)
# open_p = np.array(open_p[1100:],dtype=float)
# high   = np.array(high[1100:],dtype=float)
# low    = np.array(low[1100:],dtype=float)
# volume = np.array(volume[1100:],dtype=float)
price = np.array(price, dtype=float)
open_p = np.array(open_p, dtype=float)
high = np.array(high, dtype=float)
low = np.array(low, dtype=float)
volume = np.array(volume, dtype=float)

pprint.pprint(dataframe)


# %%
plt.figure(figsize=(6, 4), dpi=80, facecolor="w", edgecolor="k")
plt.plot(price, "b", label="Preço de fechamento")
plt.plot(open_p, "r", label="Preço de abertura")
plt.plot(high, "y", label="Preço da alta")
plt.plot(low, "g", label="Preço da baixa")
plt.ylabel("Price in R$")
plt.xlabel("Index da amostra")

plt.grid("on")
plt.legend()
plt.show()

plt.figure(figsize=(6, 4), dpi=80, facecolor="w", edgecolor="k")
plt.plot(volume, "r", label="Volume")
plt.ylabel("Volume de ações negociadas")
plt.xlabel("Index da amostra")
plt.grid("on")
plt.legend()
plt.show()


# %%
# Check if the original time series is stationary or not
price_test = np.array(price, dtype=float)
df_log = pd.DataFrame(price_test)
win_size = int(len(price_test) / 15)

# The variable of interest

plt.figure()
plt.plot(price_test, "b", label="intensity of the price_test")
plt.ylabel("Price in R$")
plt.xlabel("sample index")
plt.title("intensity of the price_test")
plt.show()

plt.figure()
plt.hist(price_test, 50)
plt.ylabel("Probability")
plt.xlabel("Price in R$")
plt.show()

# Weak stationary test
mean, cov, s_moment, std = stationary_test(price_test, delta=20, ad=True)

plt.figure()
plt.subplot()
plt.plot(price_test, "b", label="Close price")
plt.plot(mean[0], mean[1], "r", label="Rolling mean")
plt.xlabel("samples index")
plt.ylabel("Price in R$")
plt.grid("on")
plt.legend()
plt.show()

plt.figure()
plt.plot(price_test, "b", label="intensity of the price_test")
plt.plot(cov[0], cov[1], "g", label="autocovariance function")
plt.xlabel("samples index")
plt.ylabel("Covariance (ppb)")
plt.title("ACF da intensity of the price_test")
plt.legend()
plt.show()

plt.figure()
plt.plot(price_test, "b", label="intensity of the price_test")
plt.plot(mean[0], std, "g", label="standard deviation function")
plt.xlabel("samples index")
plt.ylabel("(ppb)")
plt.title("Standard deviation function")
plt.legend()
plt.show()

plt.figure()
plt.plot(price_test, "b", label="intensity of the price_test")
plt.plot(mean[0], s_moment, "g", label="Second moment function")
plt.xlabel("samples index")
plt.ylabel("(ppb)")
plt.title("Second moment function")
plt.legend()
plt.show()

# Autocorrelation function
# If intensity of the price_test is non stationary the autocorrelation plot will show periodic behavior
plt.figure()
pd.plotting.autocorrelation_plot(price_test)
plt.show()

# Based on the autocorrelation plot, is correct to infer that
# the data has a strong seasonality based on the sinusoidal
# behavior on the autocorrelation plot.
# The intensity of the price_test period is next to 1140 samples

# %% [markdown]
#   ### Analyzing the test results
#   The mean function test hypothesis failed, this can be noted by checking the
# function follows the trend of the close price, the covariation function presents a linear behavior with some disturbance, the standard deviation function seans to be linear like, but with a lot of noise.
# The second-moment function doesn't show a tendency to infinite on any $n$, that way passing on this condition.
#
# Analyzing the functions plots it's correct to infer that this realization of the close price is non-stationary, this can be also noted on the autocorrelation plot, which shows a slow decaying behavior.
# %% [markdown]
#    ### The FFT of close price could be usefull to get its harmonics components and detect a seasonal patern

# %%
y_fft = np.fft.fft(price)
y_fft_freq = np.fft.fftfreq(len(price), d=1)
spectrum_magnitude = abs(y_fft) * (2 / len(price))
end = int(len(y_fft) / 2) - 1000
plt.figure(figsize=(14, 6), dpi=80, facecolor="w", edgecolor="k")
plt.stem(y_fft_freq[:end], spectrum_magnitude[:end], "b", use_line_collection=True)
plt.xlabel("frequency (Hz)")
plt.ylabel("close price (ppb)")
plt.show()

# The DTFT of close price shows that it doesn't has a period of seasonality

# %% [markdown]
#    ## If the original time series still non-stationary a method is necessary to turn the time series stationary
#    Consider $x[n]$ as a time series, if it is non-stationary it can be divided in a seasonal, trend and a residual component.
#    $$ x[n] = s[n] + m[n] + Y[n] $$
#
#   Where $s[n]$,$m[n]$ and $Y[n]$ are respectively the seasonal, trend and residual components. The seasonal component is a function with period $d$, the trend component is a slowly changing function, the residual component is a random  noise function stationary, based on the weak stationary definition.
#
#   There are few ways to remove the trend and seasonal components, the principal ones are by estimating the $m[n]$ and $s[n]$ components, and the other one is by differencing the time series.
#
#   One way to estimate the $m[n]$ component is to apply a moving average of the time series, to remove the seasonal component a polynomial least-square regression could be done to fit a trend, then subtract the time series by fited trend.
#
#   ### Lets first analyse the differencing procedure
#    #### Differencing  a signal and invert differencing
#    1. Consider a time series $x[n]$ and a lag $p$, in order of obtaining the stationary form of the time series :
#
#    $$ diff[n] = x[n] - x[n -p] $$
#    2. Note that this process is reversible, but not to the entire length of the original $ x[n]$ time series array, is not reversible only for the first $p$ samples, that way the inverted difference array will have the length: $length(x) - p $
#
#    3. To invert the difference, the $diff[n]$ data must be added to the lagged $x[n]$ :
#
#    $$ x[n] = diff[n] + x[n-p]$$
#
#    4. The difference is a seasonal difference if $p$ is equal to $x[n]$ period
#

# %%
lag = 8
price_test = price
price_test = np.array(price_test[lag:]) - np.array(price_test[:-lag])

norm_diff_freq = price_test
plt.figure()
plt.plot(price_test, "b", label="frequency")
plt.show()

plt.figure()
plt.hist(price_test, 100)
plt.ylabel("probability")
plt.xlabel("Price in R$")
plt.show()

# Weak stationary test
mean, cov, s_moment, std = stationary_test(norm_diff_freq, delta=20, ad=True)

plt.figure()
plt.plot(norm_diff_freq, "b", label="DIFF-8ª(preço de fechamento)")
# plt.plot(mean[0], mean[1], "r", label="mean function")
plt.xlabel("Sample index")
plt.ylabel("Price in R$")
plt.legend()
plt.grid("on")
plt.show()

plt.figure()
plt.plot(norm_diff_freq, "b", label="close price")
plt.plot(cov[0], cov[1], "g", label="autocovariance function")
plt.xlabel("samples index")
plt.ylabel("Covariance (ppb)")
plt.title("ACF of close price")
plt.legend()
plt.show()

plt.figure()
plt.plot(norm_diff_freq, "b", label="close price")
plt.plot(mean[0], std, "g", label="standard deviation function")
plt.xlabel("samples index")
plt.ylabel("(ppb)")
plt.title("Standard deviation function")
plt.legend()
plt.show()

plt.figure()
plt.plot(norm_diff_freq, "b", label="close price")
plt.plot(mean[0], s_moment, "g", label="Second moment function")
plt.xlabel("samples index")
plt.ylabel("(ppb)")
plt.title("Second moment function")
plt.legend()
plt.show()

# Autocorrelation function
# If close price is non stationary the autocorrelation plot will show periodic behavior
plt.figure(figsize=(6, 4), dpi=80, facecolor="w", edgecolor="k")
# pd.plotting.autocorrelation_plot(norm_diff_freq)
pd.plotting.autocorrelation_plot(norm_diff_freq)
plt.show()


ax = plt.xcorr(norm_diff_freq, norm_diff_freq, maxlags=200)

plt.figure(figsize=(10, 4), dpi=80, facecolor="w", edgecolor="k")
lags = ax[0][200:]
corr = ax[1][200:]
plt.plot(lags, corr)
plt.ylabel("ACF")
plt.xlabel("Lags")
plt.grid(True)
plt.show()
# Based on the autocorrelation plot, is correct to infer that
# the data has a strong trend
# behavior on the autocorrelation plot.

# %% [markdown]
# ## Analysing the test results on the differencied form of close price
# The mean function is more linear than the original data, but still has strong trends. The standard deviation and covariance function also presents a linear behavior, but with a lot of noise. The autocovariance function rapdly decain behavior, with no more representative seasonal trend.
#
# From the functions plots its correct to infer that the difference of the data, using a 8 as the $d$ order , the time series is now is stationary.
# %% [markdown]
#    ### The FFT of diff close price could be usefull to get its harmonics components
#    The previous tests showed that this time series present a seasonal component, through FFT we can analyse the peaks and find out it's frequency and period.

# %%
def print_stats(mine):
    print("MIC", mine.mic())
    print("MAS", mine.mas())
    print("MEV", mine.mev())
    print("MCN (eps=0)", mine.mcn(0))
    print("MCN (eps=1-MIC)", mine.mcn_general())
    print("GMIC", mine.gmic())
    print("TIC", mine.tic())


# %% [markdown]
# ### Number of features lags to use
# As the stock price volume and close price data are time-series select

# %%
# Lag matrix
# Check the autocorrelation matrix
# Generating the autocorrelation to each lag
price = np.array(price).flatten()
volume = np.array(volume, dtype=float).flatten()
lag = 8
price_test = np.array(price[lag:]) - np.array(price[:-lag])
volume_test = np.array(volume[lag:]) - np.array(volume[:-lag])
x = np.array(price[lag:]) - np.array(price[:-lag])
x1 = x
lag = np.arange(1, 500, 1)
start = lag[-1]
end = len(x)
R = [x[start - i : end - i] for i in lag]
R1 = [x1[start - i : end - i] for i in lag]
R2 = [R1[0] for i in lag]
r = np.corrcoef(R1, R)
corr = np.abs(r[0][len(lag) :])
print("Absolute  cross correlation is bigger than 0.7 for the lags:")
print(np.where(corr > 0.8))

temp_dist_corr = np.array([dist_corr(i[0], i[1]) / 2 for i in zip(R2, R)])
print(np.where(temp_dist_corr > 0.5))

# %% [markdown]
# ### Analyzing the results
# Using the Pearson correlation coefficient perhaps doesn't represent a possible non-linear relationship between stock price volume and close price data. Another coefficient to estimate possible non-linear relationship is the Maximal information coeficient[2], this coefficient calculates the mutual information between 2 arrays.
# ### References
# [2] https://minepy.readthedocs.io/en/latest/index.html

# %%
# Mutual information
mine = MINE(alpha=0.6, c=15, est="mic_approx")
mine.compute_score(price_test, volume_test)
print_stats(mine)


# %%
# Applying the difference in all features
lag = 8
if lag > 0:
    price_test = np.array(price[lag:]) - np.array(price[:-lag])
    open_test = np.array(open_p[lag:]) - np.array(open_p[:-lag])
    low_test = np.array(low[lag:]) - np.array(low[:-lag])
    high_test = np.array(high[lag:]) - np.array(high[:-lag])
else:
    price_test = np.array(price)
    open_test = np.array(open_p)
    low_test = np.array(low)
    high_test = np.array(high)

volume_test = np.array(volume)

# %% [markdown]
# ## Normalize the data
# To normalize the training and test features we will use the equation:
#
# $$ x'_i = \frac{(x_i-x_{min})*(L_{max}-L_{min})}{x_{max}-x_{min}} + L_{min} $$
#
# where the $L_{min}$ and $L_{max}$ means the inferior and superior limits of the post normalized data. In this case we will use the interval $[0,1]$ as limits.

# %%
non_normalized_price = price_test
price_test = (price_test - np.min(price_test)) / (
    np.max(price_test) - np.min(price_test)
)
open_test = (open_test - np.min(open_test)) / (np.max(open_test) - np.min(open_test))
low_test = (low_test - np.min(low_test)) / (np.max(low_test) - np.min(low_test))
high_test = (high_test - np.min(high_test)) / (np.max(high_test) - np.min(high_test))
volume_test = (volume_test - np.min(volume_test)) / (
    np.max(volume_test) - np.min(volume_test)
)


# %%
N = 5

original_series = np.array(price[:-lag])
n_windows = len(price_test) - N

feature_mtx = np.zeros((n_windows, int(N * 5)))
label_mtx = np.zeros((n_windows, 1))
feature_lstm = list()
for i in range(0, n_windows):
    # Window start and end indexes
    i_s = i
    i_e = i + N
    # price data
    price_i = np.array(price_test[i_s:i_e])
    # open price
    open_i = np.array(open_test[i_s:i_e])
    # low price
    low_i = np.array(low_test[i_s:i_e])
    # high price
    high_i = np.array(high_test[i_s:i_e])
    # volume data
    volume_i = np.array(volume_test[i_s:i_e])
    volume_i = volume_i * np.amax(price_i) / np.amax(volume_i)

    # LSTM time dimension array
    feature_lstm_i = list()
    for j in range(0, N):
        feature_lstm_j = [
            price_test[i_s + j],
            open_test[i_s + j],
            low_test[i_s + j],
            high_test[i_s + j],
            volume_test[i_s + j],
        ]
        feature_lstm_i.append(feature_lstm_j)

    # Features array
    features = price_i
    features = np.concatenate((features, open_i))
    features = np.concatenate((features, low_i))
    features = np.concatenate((features, high_i))
    features = np.concatenate((features, volume_i))

    # LSTM features
    feature_lstm.append(feature_lstm_i)

    # Label: the first freq. offset of the next window, so that the NN is trained to predict
    label = non_normalized_price[i_e]
    # Save
    feature_mtx[i, :] = features
    label_mtx[i, :] = label

feature_lstm = np.array(feature_lstm)
# Split
x_train, x_test, y_train, y_test, i_train, i_test = train_test_split(
    feature_mtx,
    label_mtx,
    np.arange(0, n_windows),
    test_size=0.15,
    shuffle=False,
    random_state=42,
)
# Split
x_train, x_val, y_train, y_val, i_train, i_val = train_test_split(
    x_train, y_train, i_train, test_size=(15 / 85), shuffle=False, random_state=42
)

# # Split the lstm features
# xh_train, xh_test, yh_train, yh_test, ih_train, ih_test = train_test_split(
#     feature_lstm,
#     label_mtx,
#     np.arange(0, n_windows),
#     test_size=0.2,
#     shuffle=False,
#     random_state=42,
# )

# # Split
# xh_train, xh_val, yh_train, yh_val, ih_train, ih_val = train_test_split(
#     xh_train,
#     yh_train,
#     ih_train,
#     test_size=0.25,
#     shuffle=False,
#     random_state=42,
# )


# Shuffle post split data
arima_y = np.concatenate((y_train, y_val))
# x_train, y_train, i_train = shuffle(x_train, y_train, i_train, random_state=42)
# x_val, y_val, i_val = shuffle(x_val, y_val, i_val, random_state=42)


# %%
plt.figure()
plt.plot(x_train[0, :], label="First row")
plt.plot(x_train[-1, :], label="Last row")
plt.title("Training Features (one sample)")
plt.legend()

plt.figure()
plt.plot(y_train)
plt.title("Training Labels (all samples)")
plt.ylabel("ppb")


# %%
# neurons = [5,10,15,30,60]
# results = []

# for neuron in neurons :
#     results_i = []
#     for i in range(0,5):
H1 = 5
n_epochs = 500
lr = 0.001  # learning rate
lr_decay = 0.0001  # learning rate decay
n_mini_batch = 50  # mini-batch length
activation_fcn = "selu"
optimizer = Adam(lr=lr, decay=lr_decay)

model = Sequential()
model.add(Dense(input_dim=(x_train.shape[1]), units=H1, activation=activation_fcn))
# model.add(Dense(units=H2, activation=activation_fcn))

# model.add(Dense(units=H2, activation=activation_fcn))

# model.add(Dense(units=H2, activation=activation_fcn))
# model.add(Dropout(0.2))
# model.add(Dense(units=H2, activation=activation_fcn))
# model.add(Dropout(0.2))
# model.add(Dense(units=H2, activation=activation_fcn))
# model.add(Dropout(0.2))
# model.add(Dense(units=H2, activation=activation_fcn))
# model.add(Dropout(0.2))
model.add(Dense(1, activation="linear"))
model.compile(loss="mse", optimizer=optimizer)
history = model.fit(
    x_train,
    y_train,
    #     validation_data=(x_val, y_val),
    validation_split=0.2,
    batch_size=n_mini_batch,
    epochs=n_epochs,
    #     callbacks=[
    #         EarlyStopping(
    #             monitor="val_loss", mode="min", min_delta=0.01, patience=1000, verbose=1
    #         )
    #     ],
)
model.save("model.h1")
# results_i.append(history.history['val_loss'][-1])
#     results.append(results_i)


# %%


# %%
# results = np.array(results)
# mean = [np.mean(i) for i in results]

# results_df = {'Neuronios': neurons,'MSE médio de 5 repetições': mean}
# results_df = pd.DataFrame(data=results_df)
# results_df.to_csv('resultados_rede_neural.csv')
# results_df


# %%
# Plot training & validation loss values
plt.figure()
plt.plot(np.sqrt(history.history["loss"]))
plt.plot(np.sqrt(history.history["val_loss"]))
plt.title("Model loss")
plt.ylabel("sqrt(MSE)")
plt.xlabel("Epoch")
plt.legend(["Train", "Test"], loc="upper left")


# %%
# # time_dimension = False
# # if(time_dimension):
# #     xh_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))
# #     yh_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1]))
# #     xh_test  = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))
# # else:
# #     xh_train = np.reshape(x_train, (x_train.shape[0],1 ,x_train.shape[1]))
# #     yh_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1]))
# #     xh_test  = np.reshape(x_test, (x_test.shape[0],1 ,x_test.shape[1]))

# print(xh_train.shape)
# lstm = Sequential()
# # LSTM parameters
# H1             = 10
# H2             = 17
# n_epochs       = 200
# lr             = 0.001 # learning rate
# lr_decay       = 0.001    # learning rate decay
# n_mini_batch   = 40 # mini-batch length

# # Input layer
# lstm.add(LSTM(units=H1,return_sequences=True,input_shape=(xh_train.shape[1], xh_train.shape[2])))

# # Hidden layers
# # lstm.add(LSTM(units=H2, return_sequences=True))
# # lstm.add(Dropout(0.2))
# # lstm.add(LSTM(units=H2, return_sequences=True))
# # lstm.add(Dropout(0.2))
# # lstm.add(LSTM(units=H2, return_sequences=True))
# # lstm.add(Dropout(0.2))
# lstm.add(LSTM(units=H2))
# lstm.add(Dense(10,activation="selu"))
# # Output layer
# lstm.add(Dense(1,activation="linear"))

# lstm.compile(loss='mse', optimizer=Adam(lr=lr, decay=lr_decay))
# history = lstm.fit(xh_train, yh_train,
#                     validation_data=(xh_val, yh_val),
#                     batch_size=n_mini_batch,
#                     epochs=n_epochs,
#                     callbacks=[EarlyStopping(monitor='val_loss',
#                                             mode='min',
#                                             min_delta=0.01,
#                                             patience = 50,
#                                             verbose=1)])


# %%
# # Plot training & validation loss values
# plt.figure()
# plt.plot(np.sqrt(history.history["loss"]))
# plt.plot(np.sqrt(history.history["val_loss"]))
# plt.title("Model loss")
# plt.ylabel("sqrt(MSE)")
# plt.xlabel("Epoch")
# plt.legend(["Train", "Test"], loc="upper left")


# %%
# Test some predictions
yh_test = y_test.flatten()
yh_test_p = yh_test[:30]
y_nn = model.predict(x_test)
y_nn = y_nn.flatten()
y_nn_p = y_nn[:30]

n_pred = len(y_nn)

# ARIMA results

naive = np.concatenate(([0], yh_test_p[:-1]))
fig = plt.figure(figsize=(6, 4), dpi=80, facecolor="w", edgecolor="k")
plt.plot(np.arange(0, len(y_nn_p)), y_nn_p, "v-", label="Predição Rede neural")
plt.plot(np.arange(0, len(yh_test_p)), yh_test_p, "*-", label="Valor real")
plt.grid("on")
plt.legend()
plt.ylabel("Valor em R$")
plt.xlabel("Index de teste")
plt.show()
# plt.figure()
# plt.plot(i_test, y_nn, label="model")

# plt.plot(i_test, y_test.flatten(), label="Label")
# plt.legend()
# plt.ylabel("ppb")

# plt.figure()
# plt.plot(i_test, y_test.flatten(),'r' ,label="Label")
# plt.legend()
# plt.ylabel("ppb")

# plt.figure()
# plt.plot(i_test, y_nn, 'y',label="stock price volume and drift model")
# plt.legend()
# plt.ylabel("ppb")
y_nn_temp_err = y_nn_p - yh_test_p.flatten()
print(
    "[NN] Variance: %6.2f Mean: %6.2f" % (np.var(y_nn_temp_err), np.mean(y_nn_temp_err))
)
print("[NN ] MAE:       %f " % mean_absolute_error(y_nn_p, yh_test_p))
print("[NN] MSE:       %f " % mean_squared_error(y_nn_p, yh_test_p))


# %%
yh_test_p


# %%
print(y_test[0], i_test[0])
# [NN] Variance:   0.94 Mean:  -0.07
# [NN ] MAE:       0.786631
# [NN] MSE:       0.940360


# %%
y_nn_temp_err = y_nn - y_test.flatten()
# y_lstm_err    = y_lstm - y_test.flatten()
print(
    "[NN] Variance: %6.2f Mean: %6.2f" % (np.var(y_nn_temp_err), np.mean(y_nn_temp_err))
)
print("[NN ] MAE:       %f " % mean_absolute_error(y_nn, y_test))
print("[NN] MSE:       %f " % mean_squared_error(y_nn, y_test))

# print(
#     "[lstm] Variance: %6.2f Mean: %6.2f"
#     % (np.var(y_lstm_err), np.mean(y_lstm_err))
# )
# print("[lstm] MAE:       %f " % mean_absolute_error(y_lstm, y_test))
# print("[lstm] MSE:       %f " % mean_squared_error(y_lstm, y_test))

plt.figure()
plt.hist(y_nn_temp_err, bins=50, density=True)
plt.title("Histogram of NN fit error")
plt.xlabel("ppb")
plt.ylabel("Probability")

plt.show()


# %%
# ARIMA
# The holdover approach is currently a univariate timeseries
# For this purpose the ARIMA model performs well
# Split

# y_train, y_test = train_test_split(
#     wind,
#     test_size=0.4,
#     shuffle=False,
#     random_state=42,
# )

# train data autocorrelation to find the d parameter
p_order = 1
d_order = 0  # Differencing order
q_order = 0


# %%
if d_order > 0:
    train_diff = np.array(y_train[d_order:]) - np.array(y_train[:-d_order])
else:
    train_diff = y_train


fig = plt.figure()
pd.plotting.autocorrelation_plot(train_diff)

plt.show()

# The time series data used to train the model
fig = plt.figure()
plt.plot(train_diff.reshape(train_diff.size, 1))

plt.show()


# %%
arima_y = arima_y.flatten()
arima_y = arima_y.tolist()


# %%
# Defining the prediction model

arima_pred = list()


pred_scope = 400
pred_size = 30
buffer = arima_y[-1 * pred_scope :]
y_arima = []
for i in range(0, pred_size):
    arima_model = ARIMA(buffer, order=(p_order, d_order, q_order))
    arima_fit = arima_model.fit(disp=0)
    arima_pred = arima_fit.forecast(1)[0]
    buffer[: pred_scope - 1] = buffer[1:]
    buffer[-1] = y_test[i]
    y_arima.append(arima_pred)


# %%
# parameters = np.zeros((15,3))
# parameters[:5][:,0] = np.arange(1,6)
# parameters[5:10][:,0] = np.arange(1,6)
# parameters[5:10][:,2] = np.arange(1,6)
# parameters[10:][:,2] = np.arange(1,6)
# parameters = np.array(parameters,dtype=int)
# print(parameters)
# results = list()
# mse    = list()
# aic    = list()
# for parameter in parameters:
#     p_order    = parameter[0]
#     d_order    = parameter[1]
#     q_order    = parameter[2]
#     pred_scope  = 40
#     pred_size   = 1
#     buffer      = arima_y[-1*pred_scope:]
#     y_arima     = []

#     arima_model = ARIMA(buffer, order=(p_order,d_order,q_order))
#     try:
#         arima_fit   = arima_model.fit(disp=0)
#         arima_pred  = arima_fit.forecast(1)[0]
#         buffer[:pred_scope-1]   = buffer[1:]
#         buffer[-1] = y_test[i]
#         y_arima.append(arima_pred)
#         results.append(arima_fit.bic)
#         aic.append(arima_fit.aic)
#         mse.append(mean_absolute_error(y_val[0],y_arima))
#     except ValueError:
#         results.append('error')
#         mse.append('error')
#         aic.append('error')


# %%
arima_fit.aic


# %%
# parametros = ['1,0,0','2,0,0','3,0,0','4,0,0','5,0,0','1,0,1','2,0,2','3,0,3','4,0,4','5,0,5','0,0,1','0,0,2','0,0,3','0,0,4','0,0,5']
# results_df = {'Parametros':parametros,'BIC':results,'AIC':aic,'MSE':mse}
# results_df = pd.DataFrame(data=results_df)
# results_df.to_csv('parametros_arima')


# %%
plot_size = len(y_arima)
yh_test_r = y_test.flatten()
yh_test_p = yh_test_r[:plot_size]
y_nn_p = y_nn[:plot_size]
# y_lstm_p    = y_lstm[:plot_size]

# ARIMA results
y_arima = np.array(y_arima).flatten()
naive = np.concatenate(([0], yh_test_p[:-1]))
fig = plt.figure(figsize=(6, 4), dpi=80, facecolor="w", edgecolor="k")
plt.plot(np.arange(0, len(y_arima[1:])), y_arima[1:], "^-", label="Predição arima")
plt.plot(np.arange(0, len(yh_test_p[1:])), yh_test_p[1:], "*-", label="Valor real")
plt.plot(np.arange(0, len(y_nn_p[1:])), y_nn_p[1:], "v-", label="Predição Rede neural")
# plt.plot(np.arange(0, len(y_nn_p)), y_lstm_p, label="lstm")
# plt.plot(np.arange(0, len(y_nn_p[1:])), naive[1:], label="naive prediction")
plt.grid("on")
plt.legend()
plt.ylabel("Valor em R$")
plt.xlabel("Index de teste")

# Holdover error
arima_err = y_arima - yh_test_p
y_nn_temp_err = y_nn_p - yh_test_p
naive_err = naive - yh_test_p
# 100 step error
print("[ARIMA]   Variance: %6.2f Mean: %6.2f" % (np.var(arima_err), np.mean(arima_err)))
print("[ARIMA]   MAE:       %f " % mean_absolute_error(y_arima, yh_test_p))
print("[ARIMA]   MSE:       %f " % mean_squared_error(y_arima, yh_test_p))

print(
    "[NN] Variance: %6.2f Mean: %6.2f" % (np.var(y_nn_temp_err), np.mean(y_nn_temp_err))
)
print("[NN] MAE:       %f " % mean_absolute_error(y_nn_p, yh_test_p))
print("[NN] MSE:       %f " % mean_squared_error(y_nn_p, yh_test_p))


# print(
#     "[lstm] Variance: %6.2f Mean: %6.2f"
#     % (np.var(y_lstm_err), np.mean(y_lstm_err))
# )
# print("[lstm] MAE:       %f " % mean_absolute_error(y_lstm_p, yh_test_p))
# print("[lstm] MSE:       %f " % mean_squared_error(y_lstm_p, yh_test_p))
print("[Naive] Variance: %6.2f Mean: %6.2f" % (np.var(naive_err), np.mean(naive_err)))
print("[Naive] MAE:       %f " % mean_absolute_error(naive, yh_test_p))

# %% [markdown]
# ## Now the invert differencing the time series

# %%
series = price
lag = 8
y_arima_p = y_arima
split_id = i_val.max() + N + 1 + lag
past_period = series[split_id - lag : split_id]
# y_lstm_p[:lag]      = past_period + y_lstm[:lag]
y_nn_p[:lag] = past_period + y_nn[:lag]
y_arima_p[:lag] = past_period + y_arima[:lag]
yh_test_p[:lag] = past_period + yh_test_p[:lag]
true_labels = series[split_id : split_id + pred_size]

for i in range(lag, pred_size):
    past_value = series[split_id - lag + i]
    y_nn_p[i] = y_nn_p[i] + past_value
    #     y_lstm_p[i]  = y_lstm_p[i] + past_value
    y_arima_p[i] = y_arima_p[i] + past_value
    yh_test_r[i] = yh_test_r[i] + past_value


# %%
# ARIMA results
y_arima = np.array(y_arima).flatten()
naive = np.concatenate((yh_test_p[1:], [0]))
fig = plt.figure(figsize=(6, 4), dpi=80, facecolor="w", edgecolor="k")
plt.plot(np.arange(0, len(y_arima_p)), y_arima_p, "^-", label="Predição ARIMA")
plt.plot(np.arange(0, len(y_nn_p)), y_nn_p, "v-", label="Predição Rede neural")
plt.plot(np.arange(0, len(true_labels)), true_labels, "*-", label="Valor real")

plt.plot(
    np.arange(0, len(yh_test_r[:pred_size])),
    yh_test_r[:pred_size],
    "*-",
    label="labels inverted differencing",
)
# plt.plot(np.arange(0, len(y_nn_p)), naive, label="naive prediction")
plt.grid("on")
plt.legend()
plt.ylabel("Valor em R$")
plt.xlabel("Index de teste")

# plt.plot(np.arange(0, len(y_nn_p)), y_lstm_p, label="lstm")
# plt.plot(np.arange(0, len(y_nn_p[1:])), naive[1:], label="naive prediction")


# Holdover error
arima_err = y_arima_p - true_labels
y_nn_temp_err = y_nn_p - true_labels
# y_lstm_err = y_lstm_p - true_labels

# 100 step error
print("[ARIMA]   Variance: %6.2f Mean: %6.2f" % (np.var(arima_err), np.mean(arima_err)))
print("[ARIMA]   MAE:       %f " % mean_absolute_error(y_arima_p, true_labels))
print("[ARIMA]   MSE:       %f " % mean_squared_error(y_arima_p, true_labels))

print(
    "[NN] Variance: %6.2f Mean: %6.2f" % (np.var(y_nn_temp_err), np.mean(y_nn_temp_err))
)
print("[NN] MAE:       %f " % mean_absolute_error(y_nn_p, true_labels))
print("[NN] MSE:       %f " % mean_squared_error(y_nn_p, true_labels))


# print(
#     "[lstm] Variance: %6.2f Mean: %6.2f"
#     % (np.var(y_lstm_err), np.mean(y_lstm_err))
# )
# print("[lstm] MAE:       %f " % mean_absolute_error(y_lstm_p, true_labels))
# print("[lstm] MSE:       %f " % mean_squared_error(y_lstm_p, true_labels))


# %%
def get_confusion_matrix(reais, preditos, labels):
    """
    Uma função que retorna a matriz de confusão para uma classificação binária
    
    Args:
        reais (list): lista de valores reais
        preditos (list): lista de valores preditos pelo modelos
        labels (list): lista de labels a serem avaliados.
            É importante que ela esteja presente, pois usaremos ela para entender
            quem é a classe positiva e quem é a classe negativa
    
    Returns:
        Um numpy.array, no formato:
            numpy.array([
                [ tp, fp ],
                [ fn, tn ]
            ])
    """
    # não implementado
    if len(labels) > 2:
        return None

    if len(reais) != len(preditos):
        return None

    # considerando a primeira classe como a positiva, e a segunda a negativa
    true_class = labels[0]
    negative_class = labels[1]

    # valores preditos corretamente
    tp = 0
    tn = 0

    # valores preditos incorretamente
    fp = 0
    fn = 0

    for (indice, v_real) in enumerate(reais):
        v_predito = preditos[indice]

        # se trata de um valor real da classe positiva
        if v_real == true_class:
            tp += 1 if v_predito == v_real else 0
            fp += 1 if v_predito != v_real else 0
        else:
            tn += 1 if v_predito == v_real else 0
            fn += 1 if v_predito != v_real else 0

    return np.array(
        [
            # valores da classe positiva
            [tp, fp],
            # valores da classe negativa
            [fn, tn],
        ]
    )


# %%
lag = 1
true_labels_test = np.array(true_labels[lag:]) - np.array(true_labels[:-lag])
true_labels_test = np.array([1 if i > 0 else 0 for i in true_labels_test])
true_labels_test
y_arima_p_test = np.array(y_arima_p[lag:]) - np.array(true_labels[:-lag])
y_arima_p_test = np.array([1 if i > 0 else 0 for i in y_arima_p_test])
y_arima_matrix = get_confusion_matrix(
    reais=true_labels_test, preditos=y_arima_p_test, labels=[1, 0]
)
y_nn_p_test = np.array(y_nn_p[lag:]) - np.array(true_labels[:-lag])
y_nn_p_test = np.array([1 if i > 0 else 0 for i in y_nn_p_test])
y_nn_matrix = get_confusion_matrix(
    reais=true_labels_test, preditos=y_nn_p_test, labels=[1, 0]
)


# %%
print(y_nn_matrix)
print(y_arima_matrix)

from TensorAnfis.anfis import ANFIS
from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf
import time
import numpy as np
import matplotlib.pyplot as plt
import pprint


class Anfis_net:
    def __init__(self, rules=120, norm=True, set_data=False, data_csv="wine2.csv"):
        if set_data == False:
            self.load_data(data_csv)
        # else the method set dataset must be called

        # Normalize dataset
        if norm:
            print("Normalized")
            self.norm_dataset(0, 13)

        # Create ANFIS model
        if set_data == False:
            self.baseline_model(rules)

    def load_data(self, data_csv):
        # load dataset
        dataframe = pd.read_csv(data_csv, header=None)
        self.dataset = dataframe.values
        self.X = self.dataset[:, 0:13].astype(float)
        self.Y = self.dataset[:, 13:15]

    def norm_dataset(self, start, end):
        dataset = list()

        for i in range(start, end):
            norm_col = self.dataset[:, i] / np.amax(self.dataset[:, i])
            dataset.append(norm_col)

        for i in range(end, end + 2):
            col = self.dataset[:, i]
            dataset.append(col)

        dataset = pd.DataFrame(dataset)
        dataset = dataset.transpose()
        self.dataset = dataset.values
        self.X = self.dataset[:, 0:13].astype(float)
        self.Y = self.dataset[:, 13:15]

    def baseline_model(self, rules, lr=0.02):
        D = self.X.shape[1]
        self.fis = ANFIS(n_inputs=D, n_rules=rules, learning_rate=lr)

    def split_data(self, seed=42, random=True):
        if random:
            # Split data set into train and validation
            self.X_train, self.X_val, self.Y_train, self.Y_val = train_test_split(
                self.X, self.Y, test_size=0.2, random_state=seed
            )

            # Split train into train and test
            self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
                self.X_train, self.Y_train, test_size=0.25, random_state=seed
            )
            print(self.X_train.shape)

    def set_data(self, X, X_train, X_val, X_test, Y, Y_train, Y_val, Y_test):
        self.X = X
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.Y = Y
        self.Y_train = Y_train
        self.Y_val = Y_val
        self.Y_test = Y_test

    def train_and_val(self, n_epoch=900):
        print("start")
        # Train and validate the model
        with tf.Session() as sess:
            # Initialize model parameters
            sess.run(self.fis.init_variables)
            self.trn_costs = []
            self.val_costs = []
            time_start = time.time()
            for epoch in range(n_epoch):
                #  Run an update step
                trn_loss, trn_pred = self.fis.train(
                    sess, self.X_train, self.Y_train[:, 0]
                )
                # Evaluate on validation set
                val_pred, val_loss = self.fis.infer(sess, self.X_val, self.Y_val[:, 0])
                if epoch % 10 == 0:
                    print("Train cost after epoch %i: %f" % (epoch, trn_loss))
                    None
                if epoch == n_epoch - 1:
                    time_end = time.time()
                    print("Elapsed time: %f" % (time_end - time_start))
                    print("Validation loss: %f" % val_loss)
                    # Plot real vs. predicted
                self.trn_costs.append(trn_loss)
                self.val_costs.append(val_loss)
            for epoch in range(n_epoch):
                test_pred, test_loss = self.fis.infer(
                    sess, self.X_test, self.Y_test[:, 0]
                )
                if epoch == n_epoch - 1:
                    self.pred = np.vstack((np.expand_dims(test_pred, 1)))
                    self.pred = np.round(self.pred)

    def test(self):
        self.net_test = np.round([i[0] for i in self.model.predict(self.X_test)])

    def total_output(self):
        self.net_out = np.round([i[0] for i in self.model.predict(self.X)])

    def score(self):
        scores = self.model.evaluate(self.X_test, self.Y_test[:, 0])
        # evaluate the model
        print("\n%s: %.2f%%" % (self.model.metrics_names[1], scores[1] * 100))
        return scores[1]


def main():
    anfis_wine = Anfis_net()
    anfis_wine.split_data()
    anfis_wine.train_and_val()

    plt.figure(1)
    plt.plot(anfis_wine.Y_test[:, 0], "bv", label="True class")
    plt.plot(anfis_wine.pred, "r^", label="Predicted class")
    plt.xlabel("Dataset sample index")
    plt.ylabel("Corresponding class")
    plt.legend()
    plt.grid()
    # Plot the cost over epochs
    plt.figure(2)
    plt.subplot(2, 1, 1)
    plt.plot(np.squeeze(anfis_wine.trn_costs))
    plt.title("Training loss, Learning rate =")
    plt.subplot(2, 1, 2)
    plt.plot(np.squeeze(anfis_wine.val_costs))
    plt.title("Validation loss, Learning rate =")
    plt.ylabel("Cost")
    plt.xlabel("Epochs")
    # Plot resulting membership functions
    # fis.plotmfs(sess)
    plt.show()


if __name__ == "__main__":
    main()

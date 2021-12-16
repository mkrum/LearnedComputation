from collections import namedtuple
import argparse

import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


def load_data(log_file):
    lines = open(log_file).readlines()

    test_data = filter(lambda x: "test" in x, lines)
    train_data = filter(lambda x: "train" in x, lines)

    train_log = namedtuple("TrainLog", ["timestamp", "epoch", "batch_idx", "loss"])
    train_logs = []
    for t in train_data:
        timestamp, _, epoch, batch_idx, loss = t.split(",")

        tl = train_log(float(timestamp), int(epoch), int(batch_idx), float(loss))
        train_logs.append(tl)

    test_log = namedtuple("TestLog", ["timestamp", "epoch", "mse", "valid", "acc"])
    test_logs = []
    for (epoch, t) in enumerate(test_data):
        timestamp, _, mse, valid, acc = t.split(",")
        tl = test_log(float(timestamp), epoch, float(mse), float(valid), float(acc))
        test_logs.append(tl)

    return train_logs, test_logs


def loss_plot(ax, base_timestamp, train_logs):
    timestamps = list(map(lambda x: x.timestamp - base_timestamp, train_logs))
    train_losses = list(map(lambda x: x.loss, train_logs))

    ax.plot(timestamps, train_losses)
    ax.set_title("Training Loss")
    ax.set_ylabel("Negative Log Likelihood")


def acc_and_valid_plot(ax, base_timestamp, test_logs):
    timestamps = list(map(lambda x: x.timestamp - base_timestamp, test_logs))
    accs = list(map(lambda x: 100 * x.acc, test_logs))
    valids = list(map(lambda x: 100 * x.valid, test_logs))

    ax.plot(timestamps, accs)
    ax.plot(timestamps, valids)
    ax.set_ylim([0, 100])
    ax.set_ylabel("Percentage")
    ax.legend(["Accuracy", "Validity"])
    ax.set_title("Success Metrics")


def mse_plot(ax, base_timestamp, test_logs):
    timestamps = list(map(lambda x: x.timestamp - base_timestamp, test_logs))
    mse = list(map(lambda x: x.mse, test_logs))
    ax.set_yscale("log")
    ax.plot(timestamps, mse)
    ax.set_title("MSE")
    ax.set_ylabel("MSE")


def training_plot(log_file):
    train_logs, test_logs = load_data(log_file)
    base_timestamp = test_logs[0].timestamp

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(3, 9))

    loss_plot(ax1, base_timestamp, train_logs)
    acc_and_valid_plot(ax2, base_timestamp, test_logs)
    mse_plot(ax3, base_timestamp, test_logs)
    ax3.set_xlabel("Time (s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("log_file")

    args = parser.parse_args()
    training_plot(args.log_file)
    plt.show()

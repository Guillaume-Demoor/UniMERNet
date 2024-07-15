import csv
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def main(csv_path):
    data = pd.read_csv(csv_path)
    for column in data.columns:
        if column != "Epoch":
            plt.plot(data["Epoch"], data[column], label=column)
    plt.xlabel("Epoch")
    if data.shape[1] == 4:
        plt.ylabel("Metrics on validation")
    if data.shape[1] == 2:
        plt.ylabel("Training loss")
    plt.legend()
    sns.set_style("whitegrid")
    plt.show()


path = "/home/gdemoor/warm/TestGuill/UniMERNet/UniMERNet/metrics.csv"
main(path)
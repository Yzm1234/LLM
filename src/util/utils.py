import matplotlib.pyplot as plt
from datasets import load_dataset
from datasets import load_dataset,Features,Value
from datasets import load_from_disk


def split_dataset(data_path: str, output_file: str, test_size: float=0.2, random_seed: int=42):
    """
    split the datset into train and test set
    :param data_path: full dataset
    :type data_path: string
    :param output_file: output of splitted dataset
    :type output_file: string
    :param test_size: ration of test size
    :type test_size: float number
    :param random_seed: random seed used to split
    :type random_seed: int
    :return: splitted dataset
    :rtype: DatasetDict
    """
    ds = load_dataset("json", data_files=data_path, split='train', field='data')
    ds = ds.train_test_split(test_size=test_size, seed=random_seed)
    ds.save_to_disk(output_file)
    print(f"Splitted dataset is saved to {output_file}.")
    return ds


def get_tuning_loss_plot(log_hist, graph_name):
    epochs, train_loss, val_loss = [], [], []
    for i in range(len(log_hist)//2):
        epochs.append(log_hist[2*i]['epoch'])
        train_loss.append(log_hist[2*i]['loss'])
        val_loss.append(log_hist[2*i+1]['eval_loss'])

    plt.plot(epochs, train_loss, label="train loss")
    plt.plot(epochs, val_loss, label="validation loss")
    plt.legend()
    plt.title("Model fine tune loss")
    plt.xlabel("steps")
    plt.ylabel("loss")
    plt.savefig(f"{graph_name}_tuning_loss.png")
    plt.show()
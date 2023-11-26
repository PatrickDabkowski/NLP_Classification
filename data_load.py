from torchtext import datasets
from torch.utils.data import DataLoader, random_split

def load(valid=True, batch_size=16):
    print("dataset loading...")
    train_iter = datasets.IMDB(split='train')
    test_set = datasets.IMDB(split='test')

    if valid:

        test_data = list(test_set)
        split_index = len(test_data) // 2
        valid_set, test_set = random_split(test_data, [split_index, len(test_data) - split_index])

        valid_iter = DataLoader(valid_set, batch_size=batch_size)
        test_iter = DataLoader(test_set, batch_size=batch_size)

        print("train, validation, test loaded...")
        return train_iter, valid_iter, test_iter

    else:

        print("train, test loaded...")
        return train_iter, test_set

if __name__ != "__main__":
    print(f"file: {__name__}")
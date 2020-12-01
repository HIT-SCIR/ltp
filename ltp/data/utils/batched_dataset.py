from torch.utils.data import Dataset, DataLoader


class BatchedDataset(Dataset):
    def __init__(self, batches):
        self.batches = batches

    def __getitem__(self, index):
        return self.batches[index]

    def __len__(self):
        return len(self.batches)


class BatchedDataloader(DataLoader):
    pass

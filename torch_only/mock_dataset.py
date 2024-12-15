import torch
from torch.utils.data import Dataset
from torch.utils.data import IterableDataset


class MockDataset(Dataset):
    def __init__(self, seq_len: int=32, vocab_size: int=32_000, n_samples=1_000, seed_offset=0):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.n_samples = n_samples
        self.seed_offset = seed_offset
        self.gen = torch.Generator()

    def __len__(self):
        return self.n_samples    

    def __getitem__(self, idx):
        if idx >= self.n_samples:
            raise ValueError()
        self.gen.manual_seed(idx + self.seed_offset)
        return torch.randint(low=0, high=self.vocab_size, size=(1, self.seq_len), generator=self.gen)


class MockIterableDataset(IterableDataset):
    def __init__(self, seq_len: int=32, vocab_size: int=32_000, n_samples=1_000, seed_offset=0):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.n_samples = n_samples
        self.seed_offset = seed_offset
        self.gen = torch.Generator()

    def __len__(self):
        return self.n_samples    

    def __iter__(self):
        for idx in range(self.n_samples):
            self.gen.manual_seed(idx + self.seed_offset)
            yield torch.randint(low=0, high=self.vocab_size, size=(1, self.seq_len), generator=self.gen)
import itertools
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from dataclasses import dataclass
from connect_four.connect_four import generate_random_game, ConnectFour


class ConnectFourDataset:
    def __init__(
            self,
            data_size: int,
            train_size: int,
            rows_count: int = 6,
            columns_count: int = 7,
            games_to_use: list[int] = None,
        ):
        if games_to_use is None:
            games_to_use = [
                generate_random_game(
                    ConnectFour(rows_count=rows_count, columns_count=columns_count)
                )[1].history
                for _ in tqdm(range(data_size))
            ]
        self.sequences = games_to_use[:train_size]
        self.valid = games_to_use[train_size:]

    def __len__(self, ):
        return len(self.sequences)
    
    def __getitem__(self, i):
        return self.sequences[i]
    


@dataclass
class DatasetPreprocessingConfig:
    """
    Stores:
    to_model_repr: mapping used to map the transcripts into the format accepted by the model,
    from_model_repr: mapping used to translate the model's output to the original format
        of the game transcription.
    data_size: used by the GPT
    vocab_size: used by the GPT
    """
    to_model_repr: dict[int, int]
    from_model_repr: dict[int, int]
    block_size: int
    vocab_size: int


class CharConnectFourDataset(Dataset):
    """
    Dataloader for the model. Based on the OthelloGPT implementation.
    """
    def __init__(
            self,
            cf_dataset: ConnectFourDataset
    ):
        self.chars = sorted(list(set(list(itertools.chain.from_iterable(cf_dataset)))) + [-100, ])
        data_size, vocab_size = len(cf_dataset), len(self.chars)  # vocab size 61, with -100 sorted to the front
        max_len = max([len(cf_dataset[i]) for i in range(len(cf_dataset))])  # should be 60 in Othello
        print('Dataset created has %d sequences, %d unique words.' % (data_size, vocab_size))
        to_model_repr = {ch: i for i, ch in enumerate(self.chars)} # "stoi"
        from_model_repr = {i: ch for i, ch in enumerate(self.chars)} # "itos"
        self.config = DatasetPreprocessingConfig(
            to_model_repr=to_model_repr,
            from_model_repr=from_model_repr,
            block_size=max_len - 1,
            vocab_size=vocab_size
        )
        self.max_len = max_len
        self.data = cf_dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx]
        if len(chunk) != self.max_len:
            # if the data sample is smaller than max_len, fill the end with -100
            # Those -100 will be translated to 0 for the model
            chunk += [-100, ] * (self.max_len - len(chunk))  # -100 can be ignored in CE
        # encode every character to an integer
        dix = [self.config.to_model_repr[s] for s in chunk]
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y
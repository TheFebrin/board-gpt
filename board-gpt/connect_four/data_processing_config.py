from dataclasses import dataclass


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

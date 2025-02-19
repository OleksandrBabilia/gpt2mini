from dataclasses import dataclass


@dataclass
class GPTConfigLite:
    block_size: int = 256
    vocab_size: int = 65 
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384


@dataclass
class GPTConfig:
    block_size: int = 1024 
    vocab_size: int = 50257 
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

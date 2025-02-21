import torch
import tiktoken


class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, data_path="../data/input.txt"):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes

        with open(data_path, "r") as file:
            text = file.read()
        
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)

        print(f"Loaded {len(self.tokens)} tokens")
        print(f"1 epoch {len(self.tokens) // (B*T)} batches")

        self.current_position = self.B * self.T * self.process_rank
    
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position: self.current_position + B * T + 1]
        
        x = (buf[:-1].view(self.B, self.T))
        y = (buf[1:].view(self.B, self.T))

        self.current_position += self.B * self.T * self.num_processes

        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_position = self.B * self.T * self.process_rank
        
        return x, y
import torch
from torch.nn import functional as F

from configs.gptconfig import GPTConfig 
from models.gpt import GPT
from utils.dataloader import DataLoaderLite
 
if __name__ == "__main__":
    device = "cpu"
    if torch.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    print(f"Using {device}")
    model = GPT(GPTConfig())
    model.to(device)
    print("Model loaded")

    num_return_sequences = 5
    max_length = 30

    train_loader = DataLoaderLite(B=4, T=32)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    for i in range(50):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits, loss = model(x, y)
        loss.backward()
        optimizer.step()
        print(f"step {i}, loss {loss.item()}")
    
    import sys; sys.exit(0)

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    model.eval()
    while x.size(1) < max_length:
        with torch.no_grad():
            logits = model(x)
            logits = logits[:, -1, :]

            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            ix = torch.multinomial(topk_probs, 1)

            xcol = torch.gather(topk_indices, -1, ix)
            x = torch.cat((x, xcol), dim=1)
    
    for i in range(num_return_sequences):
        token = x[i, :max_length].tolist()
        decode = enc.decode(token)
        print(">", decode)



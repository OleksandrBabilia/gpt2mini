import torch
from torch.nn import functional as F
import tiktoken

from configs.gptconfig import GPTConfig 
from models.parts import GPT
 
if __name__ == "__main__":
    # model = GPT.from_pretrained("gpt2")
    model = GPT(GPTConfig())
    model.to("cuda")
    print("model loaded")

    num_return_sequences = 5
    max_length = 30

    enc = tiktoken.get_encoding("gpt2")
    with open("../data/input.txt") as f:
        text = f.read()
    text = text[:1000]
    tokens = enc.encode(text)

    B, T = 4, 32
    buf = torch.tensor(tokens[:B*T + 1])
    buf = buf.to("cuda")
    x = buf[:-1].view(B, T)
    y = buf[1:].view(B, T)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    for i in range(50):
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



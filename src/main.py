import torch
from torch.nn import functional as F
import time

from configs.gptconfig import GPTConfig 
from models.gpt import GPT
from utils.dataloader import DataLoaderLite
from utils.schedular import get_lr

if __name__ == "__main__":
    device = "cpu"
    if torch.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"

    print(f"Using {device}")
    model = GPT(GPTConfig())
    model.to(device)
    model = torch.compile(model)
    print("Model loaded")

    num_return_sequences = 5
    max_length = 30
    total_batch_size = 524288 
    B = 2
    T = 1024
    assert total_batch_size % (B * T) == 0, "make sure total_batch_size is divisible by B * T"
    grad_accum_step = total_batch_size // (B * T)
    print(f"Total desired batch size: {total_batch_size}")
    print(f"=> calulated gradient accumulation steps: {grad_accum_step}")

    train_loader = DataLoaderLite(B=2, T=1024)
    torch.set_float32_matmul_precision("high")
    optimizer = model.configure_optimizer(weight_decay=0.1, learning_rate=6e-4, device=device)

    for step in range(50):
        t0 = time.time()
        optimizer.zero_grad()
        loss_acum = 0.0
        for mirc_step in range(grad_accum_step):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)

            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            loss = loss / grad_accum_step
            loss_acum += loss.detach()
            loss.backward()

        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        optimizer.step()
        torch.cuda.synchronize()

        t1 = time.time()
        dt = (t1 - t0) * 1000
        tokens_per_second = (train_loader.B * train_loader.T * grad_accum_step) / (t1 - t0)
        print(f"step {step} | loss {loss_acum.item():.4f} | norm: {norm:.4f} | lr {lr:.4f}| dt: {dt:.2f}ms | tok/sec: {tokens_per_second:.2f}")
    
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



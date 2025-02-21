import torch
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import time
import os

from configs.gptconfig import GPTConfig 
from models.gpt import GPT
from utils.dataloader import DataLoaderLite
from utils.schedular import get_lr

if __name__ == "__main__":
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        assert torch.cuda.is_available(), "No CUDA availible"
        dist.init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        device = "cpu"
        if torch.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        print(f"Using {device}")
    model = GPT(GPTConfig())
    model.to(device)
    model = torch.compile(model)
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model

    num_return_sequences = 5
    max_length = 30
    total_batch_size = 524288 
    B = 2
    T = 1024
    assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
    grad_accum_step = total_batch_size // (B * T * ddp_world_size)
    if master_process:
        print(f"Total desired batch size: {total_batch_size}")
        print(f"=> calulated gradient accumulation steps: {grad_accum_step}")

    train_loader = DataLoaderLite(B=2, T=1024, process_rank=ddp_rank, num_processes=ddp_world_size)
    torch.set_float32_matmul_precision("high")
    optimizer = raw_model.configure_optimizer(weight_decay=0.1, learning_rate=6e-4, device=device)

    for step in range(50):
        t0 = time.time()
        optimizer.zero_grad()
        loss_acum = 0.0
        for micro_step in range(grad_accum_step):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)

            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            loss = loss / grad_accum_step
            loss_acum += loss.detach()
            if ddp:
                model.require_backward_grad_sync = (micro_step == grad_accum_step - 1)
            loss.backward()
        if ddp:
            dist.all_reduce(loss_acum, op=dist.ReduceOp.AVG)

        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        optimizer.step()
        torch.cuda.synchronize()

        t1 = time.time()
        dt = (t1 - t0) * 1000
        tokens_per_second = (train_loader.B * train_loader.T * grad_accum_step * ddp_world_size) / (t1 - t0)
        print(f"step {step} | loss {loss_acum.item():.4f} | norm: {norm:.4f} | lr {lr:.4f}| dt: {dt:.2f}ms | tok/sec: {tokens_per_second:.2f}")
    
    if ddp:
        dist.destroy_process_group()

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



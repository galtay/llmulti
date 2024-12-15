"""
https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
https://github.com/pytorch/examples/tree/main/distributed/ddp-tutorial-series
"""

import argparse

from datasets import load_dataset
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
import wandb


from model_def import GPTClone


torch.set_float32_matmul_precision("high")


class Trainer:

    def __init__(
        self,
        model: torch.nn.Module,
        train_dl: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int,
        log_every: int,
        bf16_mixed: bool,
        lr_scheduler: optim.lr_scheduler.LRScheduler,
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_dl = train_dl
        self.optimizer = optimizer
        self.save_every = save_every
        self.log_every = log_every
        self.bf16_mixed = bf16_mixed
        self.lr_scheduler = lr_scheduler

    def _run_batch(self, source, targets):
        if self.bf16_mixed:
            scaler = GradScaler()
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = self.model(source)
                loss = F.cross_entropy(
                    outputs.view(-1, outputs.size(-1)), targets.view(-1)
                )
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
            self.optimizer.zero_grad()
        else:
            outputs = self.model(source)
            loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        self.lr_scheduler.step()
        return loss.item()

    def _run_epoch(self, epoch):
        b_sz = next(iter(self.train_dl)).shape[0]
        print(
            f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_dl)}"
        )
        loss_buffer = []
        for step, batch in enumerate(tqdm(self.train_dl)):
            source = batch[:, :-1].to(self.gpu_id)
            targets = batch[:, 1:].to(self.gpu_id)
            loss = self._run_batch(source, targets)
            loss_buffer.append(loss)
            if step % self.log_every == 0:
                avg_loss = sum(loss_buffer) / len(loss_buffer)
                if self.gpu_id == 0:
                    current_lr = self.lr_scheduler.get_last_lr()[0]
                    wandb.log(
                        data={
                            "train-loss": avg_loss,
                            "learning_rate": current_lr,
                        },
                        step=step,
                    )
                loss_buffer = []
            if step % self.save_every == 0 and step != 0 and self.gpu_id == 0:
                self._save_checkpoint(step)

    def _save_checkpoint(self, step):
        ckp = self.model.state_dict()
        path = "checkpoint.pt"
        torch.save(ckp, path)
        print(f"Step {step} | Training checkpoint saved at {path}")

    def train(self, max_epochs: int):
        self.model.train()
        for epoch in range(max_epochs):
            self._run_epoch(epoch)


def get_bert_tokenizer():
    tokenizer_name = "google-bert/bert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return tokenizer


class SimpleDataset(Dataset):

    def __init__(self, local):
        self.local = local
        self.df = pd.read_parquet(self.local)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(
            self.df.iloc[idx]["tokens"].copy(),
        ).to(torch.int64)


class HfDataset(Dataset):

    def __init__(self, name, split):
        self.name = name
        self.split = split
        self.ds = load_dataset(name, split=split)
        self.ds.set_format(type="torch", columns=["tokens"])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return self.ds[idx]["tokens"]


def load_train_objs(args):
    # train_ds = SimpleDataset(local="smollm-corpus-mini-val.parquet")
    # train_ds = SimpleDataset(local="smollm-corpus-mini-train.parquet")
    train_ds = HfDataset("gabrielaltay/smollm-mini", "validation")

    tokenizer = get_bert_tokenizer()
    vocab_size = len(tokenizer)
    d_model = 768
    feedforward_dim = d_model * 4
    n_heads = 12
    num_layers = 6
    dropout = 0.1
    seq_length = 1024
    model = GPTClone(
        vocab_size, d_model, n_heads, num_layers, feedforward_dim, dropout, seq_length
    )
    optimizer = optim.AdamW(model.parameters(), lr=args.lr_max)

    return train_ds, model, optimizer


def main(args):
    torch.manual_seed(args.torch_seed)
    wandb.init(project=args.wandb_project, config=vars(args))
    train_ds, model, optimizer = load_train_objs(args)
    model = torch.compile(model)
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        shuffle=False,
    )
    total_steps = len(train_dl) * args.max_epochs
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr_max,
        total_steps=total_steps,
        pct_start=args.lr_pct_start,
        final_div_factor=args.lr_final_div,
    )
    trainer = Trainer(
        model,
        train_dl,
        optimizer,
        args.device,
        args.save_every,
        args.log_every,
        args.bf16_mixed,
        lr_scheduler=lr_scheduler,
    )
    trainer.train(args.max_epochs)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0, help="cuda:[device]")
    parser.add_argument(
        "--bf16_mixed", default=True, action=argparse.BooleanOptionalAction
    )
    parser.add_argument(
        "--max_epochs", type=int, default=1, help="total epochs to train the model"
    )
    parser.add_argument(
        "--save_every", type=int, default=512, help="checkpoint after this many steps"
    )
    parser.add_argument(
        "--log_every", type=int, default=8, help="log after this many steps"
    )
    parser.add_argument(
        "--lr_max", type=float, default=2e-4, help="maximum learning rate"
    )
    parser.add_argument(
        "--lr_pct_start",
        type=float,
        default=5e-2,
        help="percent of total time for warmup",
    )
    parser.add_argument(
        "--lr_final_div",
        type=float,
        default=1e1,
        help="lr_max / lr_final_div = final learning rate",
    )
    parser.add_argument(
        "--batch_size", default=32, type=int, help="input batch size on each device"
    )
    parser.add_argument(
        "--torch_seed", default=1337, type=int, help="seed for torch.manual_seed"
    )
    parser.add_argument(
        "--wandb_project", default="llmulti", type=str, help="wandb project name"
    )
    args = parser.parse_args()
    print(args)
    main(args)

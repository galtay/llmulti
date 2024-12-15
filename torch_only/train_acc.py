from datasets import load_dataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
import wandb
from accelerate import Accelerator

from archive.constant_length_dataset import ConstantLengthDataset


torch.set_float32_matmul_precision('high')


class GPTClone(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        num_heads,
        num_layers,
        feedforward_dim,
        dropout,
        max_seq_len,
    ):
        super(GPTClone, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output_layer = nn.Linear(embed_dim, vocab_size)

        
    def forward(self, idx, targets=None):
        batch_size, seq_len = idx.shape
        positions = torch.arange(0, seq_len, device=idx.device).unsqueeze(0).expand(batch_size, seq_len)

        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(positions)
        x = tok_emb + pos_emb

        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len)        
        logits = self.output_layer(self.transformer(x, mask=causal_mask, is_causal=True))
        if targets is not None:
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        else:
            loss = None
            
        return logits, loss

# Training Loop
def train(model, dataloader, criterion, optimizer):

    loss_buffer = []
    total_loss = 0
    for step, batch in enumerate(dataloader):

        inputs = batch[:, :-1]
        targets = batch[:, 1:]
        optimizer.zero_grad()
        
        logits, loss = model(inputs, targets=targets)
        accelerator.backward(loss)
        optimizer.step()
        total_loss += loss.item()
        loss_buffer.append(loss.item())

        if step == 0:
            if accelerator.is_local_main_process:
                print("inputs.shape", inputs.shape)
                print("targets.shape", targets.shape)
                print("logits.shape", logits.shape)
                
        
        if step % 10 == 0:
            avg_loss = sum(loss_buffer) / len(loss_buffer)
            accelerator.log({"train-loss": avg_loss}, step=step)
            loss_buffer = []
            if accelerator.is_local_main_process:
                print("train_loss", step, avg_loss)
            
    return total_loss / len(dataloader)


# Evaluation Loop
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch[:, :-1].to(device)
            targets = batch[:, 1:].to(device)
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            total_loss += loss.item()
    return total_loss / len(dataloader)


def get_llama_32_tokenizer():
    """
    128004: AddedToken(
        "<|finetune_right_pad_id|>",
        rstrip=False, lstrip=False, single_word=False, normalized=False, special=True
    )
    """
    tokenizer_name = "meta-llama/Llama-3.2-1B"
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        pad_token = "<|finetune_right_pad_id|>",        
    )
    return tokenizer


def get_gpt2_tokenizer():
    tokenizer_name = "openai-community/gpt2"
    tokenizer_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return tokenizer


if __name__ == "__main__":

    epochs = 1
    learning_rate = 5e-5

    d_model = 768
    n_heads = 12
    num_layers = 6
    dropout = 0.1
    
    accelerator = Accelerator(
        log_with="wandb",
        mixed_precision="bf16",
    )
    accelerator.init_trackers(
        project_name="torchey", 
        config={
            "dropout": dropout,
            "learning_rate": learning_rate,
            "d_model": d_model,
            "n_heads": n_heads,
            "num_layers": num_layers,
        }
    )
    
    device = accelerator.device
    
    tokenizer = get_gpt2_tokenizer()    
    wiki_dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train")
    seq_length=8192
    num_of_sequences=8
    batch_size = 2
    
    dataset = ConstantLengthDataset(
        tokenizer,
        wiki_dataset,
        seq_length=seq_length,
        num_of_sequences=num_of_sequences,
        concat_token_id = tokenizer.convert_tokens_to_ids(tokenizer.special_tokens_map['eos_token'])
    )
    dataloader = DataLoader(dataset, batch_size=batch_size)

    vocab_size = len(tokenizer)


    # Initialize the model
    model = GPTClone(
        vocab_size,
        d_model,
        n_heads,
        num_layers,
        d_model * 4,
        dropout,
        seq_length
    )

    model = model.to(device)
    
    model = torch.compile(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    device = accelerator.device
    model, optimizer, dataloader = accelerator.prepare(
        model, optimizer, dataloader
    )

    if accelerator.is_local_main_process:
        wandb.init(
            project="torchey",
        )
    
    # Training and Evaluation
    for epoch in range(epochs):
        train_loss = train(model, dataloader, criterion, optimizer)
#        train_loss = train_with_mixed_precision(model, dataloader, criterion, optimizer, device)        
#        val_loss = evaluate(model, dataloader, criterion, device)
        val_loss = 0
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")


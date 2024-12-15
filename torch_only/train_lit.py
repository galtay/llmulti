"""
https://docs.wandb.ai/guides/integrations/lightning/

"""

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer
from streaming import StreamingDataLoader
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy

from text_dataset import StreamingTextDataset


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

    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0).expand(x.size(0), seq_len)
        x = self.token_embedding(x) + self.position_embedding(positions)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)        
        x = self.transformer(x, mask=causal_mask, is_causal=True)
        return self.output_layer(x)


class LitGPTClone(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def training_step(self, batch, batch_idx):
        inputs = batch[:, :-1]
        targets = batch[:, 1:]
        outputs = self.model(inputs)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs.view(-1, outputs.size(-1)), targets.reshape(-1))        
        self.log("train_loss", loss)
        return loss
    
    def configure_optimizers(self):
        learning_rate = 5e-5
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        return optimizer 


def get_bert_tokenizer():
    tokenizer_name = "google-bert/bert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return tokenizer


if __name__ == "__main__":
    L.seed_everything(83674)
    tokenizer = get_bert_tokenizer()
    
    seq_length=1024
    batch_size = 16
    dataset = StreamingTextDataset(
        local="smollm-corpus-mini-val",
        tokenizer=tokenizer,
        max_seq_len=seq_length,
        batch_size=batch_size,
        shuffle=True,
    )
    dataloader = StreamingDataLoader(dataset, batch_size=batch_size)   
    epochs = 1
    
    vocab_size = len(tokenizer)
    d_model = 768
    n_heads = 12
    num_layers = 6
    dropout = 0.1

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

    lit_model = LitGPTClone(model)
    wandb_logger = WandbLogger(log_model="all")

    #ddp = DDPStrategy(process_group_backend="nccl")
    ddp = DDPStrategy(process_group_backend="gloo")
    trainer = L.Trainer(
        limit_train_batches=None, 
        max_epochs=epochs, 
        logger=wandb_logger,
        log_every_n_steps=1,
        precision='bf16-mixed',
        devices=[0,1],
        strategy='ddp_spawn',
    )
    trainer.fit(model=lit_model, train_dataloaders=dataloader)



    

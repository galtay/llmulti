"""
one file
https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus/blob/main/fineweb-edu-dedup/train-00000-of-00234.parquet

Note loading strategy
https://github.com/mosaicml/llm-foundry/blob/main/llmfoundry/data/text_data.py

"""

from huggingface_hub import HfApi
import numpy as np
import pandas as pd
from streaming import MDSWriter
from streaming import StreamingDataset
from transformers import AutoTokenizer
from tqdm import tqdm


def write_mds_split(out_root, df):
    name = "google-bert/bert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(name)
    vocab_size = len(tokenizer)
    print(f"{vocab_size=}")
    if vocab_size < np.iinfo(np.int16).max:
        print("vocab fits in int16")
        columns = {"tokens": "ndarray:int16"}
    elif vocab_size < np.iinfo(np.int32).max:
        print("vocab fits in int32")
        columns = {"tokens": "ndarray:int32"}
    elif vocab_size < np.iinfo(np.int64).max:
        print("vocab fits in int64")
        columns = {"tokens": "ndarray:int64"}

    compression = None
    bos_token_id = tokenizer.convert_tokens_to_ids(
        tokenizer.special_tokens_map["cls_token"]
    )
    eos_token_id = tokenizer.convert_tokens_to_ids(
        tokenizer.special_tokens_map["sep_token"]
    )

    seq_len = 1024

    with MDSWriter(out=out_root, columns=columns, compression=compression) as out:
        buf = []
        for ii, sample in tqdm(df.iterrows(), total=df.shape[0]):
            input_ids = tokenizer(sample["text"], add_special_tokens=False)["input_ids"]
            input_ids = [bos_token_id] + input_ids + [eos_token_id]
            buf.extend(input_ids)

            while len(buf) >= seq_len + 1:
                samp = buf[: seq_len + 1]
                buf = buf[seq_len + 1 :]
                x = np.array(samp, dtype=np.int16)
                out.write({"tokens": x})


upload_to_hf = True
df = pd.read_parquet("/home/galtay/data/sdb/smollm-corpus/train-00000-of-00234.parquet")
ff = int(df.shape[0] * 0.95)
df_train = df.iloc[:ff]
df_val = df.iloc[ff:]

out_root = "smollm-corpus-mini-val"
write_mds_split(out_root, df_val)
ds = StreamingDataset(local=out_root, batch_size=512)
df_tmp = pd.DataFrame(ds)
df_tmp.to_parquet(f"{out_root}.parquet")

out_root = "smollm-corpus-mini-train"
write_mds_split(out_root, df_train)
ds = StreamingDataset(local=out_root, batch_size=512)
df_tmp = pd.DataFrame(ds)
df_tmp.to_parquet(f"{out_root}.parquet")


if upload_to_hf:
    api = HfApi()
    api.upload_file(
        path_or_fileobj="smollm-corpus-mini-val.parquet",
        path_in_repo="validation.parquet",
        repo_id="gabrielaltay/smollm-mini",
        repo_type="dataset",
    )
    api.upload_file(
        path_or_fileobj="smollm-corpus-mini-train.parquet",
        path_in_repo="train.parquet",
        repo_id="gabrielaltay/smollm-mini",
        repo_type="dataset",
    )

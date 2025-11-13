"""
1. 通过pdbs和summary创建Complex实例
2. 获取heavy light antigen chains，其中antigen只关注其中最近的50个
3.
"""

#!/usr/bin/python
# -*- coding:utf-8 -*-
import os

from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

from byprot import utils
from byprot.datamodules.dataset.tokenized_protein import (
    ApproxBatchSampler,
    DPLM2Collater,
    DPLM2Tokenizer,
    SortishSampler,
)

log = utils.get_logger(__name__)


class TokenizedComplexDataset(Dataset):
    """
    Dataset that pulls from SAbDab downloads.
    """

    def __init__(
        self,
        data_dir: str,
        split: str,
        csv_file: str,
        max_len=2048,
        vocab_file="airkingbd/dplm2_650m",
        struct_vocab_size=8192,
    ):
        self.data_dir = data_dir
        self.split = split
        data_path = os.path.join(self.data_dir, csv_file.replace(".csv", ""))
        self.data = load_dataset(data_path, name=split)["train"]  # type: ignore
        log.info(f"Dataset size: {len(self.data)}")

        self.max_len = max_len
        self.tokenizer = DPLM2Tokenizer.from_pretrained(vocab_file)

    def __len__(self):
        return len(self.data)

    def get_metadata_lens(self):
        return self.data["length"]  # type: ignore

    def __getitem__(self, idx):
        row = self.data[int(idx)]
        struct_tokens = (
            self.tokenizer.struct_cls_token
            + "".join(row["heavy_struct_tok_seq"].split(","))
            + self.tokenizer.struct_eos_token
            + self.tokenizer.struct_cls_token
            + "".join(row["light_struct_tok_seq"].split(","))
            + self.tokenizer.struct_eos_token
            + self.tokenizer.struct_cls_token
            + "".join(row["epitope_struct_tok_seq"].split(","))
            + self.tokenizer.struct_eos_token
        )
        aatype_tokens = (
            self.tokenizer.aa_cls_token
            + row["heavy_aa_tok_seq"]
            + self.tokenizer.aa_eos_token
            + self.tokenizer.aa_cls_token
            + row["light_aa_tok_seq"]
            + self.tokenizer.aa_eos_token
            + self.tokenizer.aa_cls_token
            + row["epitope_aa_tok_seq"]
            + self.tokenizer.aa_eos_token
        )

        return_dict = {
            "struct_tokens": struct_tokens,
            "aatype_tokens": aatype_tokens,
            "length": len(struct_tokens) + 2,
        }
        if "pdb_name" in row:
            return_dict["pdb_name"] = row["pdb_name"]

        return return_dict


# TODO: refactor the function for complex
def setup_dataloader(
    ds: TokenizedComplexDataset,
    max_tokens=6000,
    bucket_size=1000,
    max_batch_size=100,
    num_workers=8,
    rank=0,
    world_size=1,
    max_len=512,
    tokenizer=None,
    epoch=0,
) -> DataLoader:
    collater = DPLM2Collater(tokenizer)
    lens = ds.get_metadata_lens()
    train_sortish_sampler = SortishSampler(
        lens, bucket_size, num_replicas=world_size, rank=rank, epoch=epoch
    )
    train_sampler = ApproxBatchSampler(
        train_sortish_sampler,
        max_tokens,
        max_batch_size,
        lens,
        max_len=max_len,
    )
    dl = DataLoader(
        dataset=ds,
        batch_sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=collater,
    )
    return dl

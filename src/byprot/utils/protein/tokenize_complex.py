# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0


import argparse
import hashlib
import json
import os
import warnings

import torch
from biotite.sequence.io import fasta
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from byprot.datamodules.pdb_dataset import utils as du
from byprot.datamodules.pdb_dataset.pdb_datamodule import (
    PdbDataset,
)
from byprot.models.structok.structok_lfq import VQModel
from byprot.models.utils import get_struct_tokenizer
from byprot.utils import get_logger, recursive_to

warnings.filterwarnings("ignore")

torch.set_float32_matmul_precision("high")
log = get_logger(__name__)


def chains2label(pdb_id: str, H: str, L: str, A: "list[str]"):
    if H == L.lower() or H == L.upper():
        # TODO: NonaAntibody Implementation
        return ""
    return ".".join([pdb_id, H, L, "".join(sorted(A))])


def load_from_pdb(
    pdb_path, process_chain=PdbDataset.process_chain, summary: dict = {}
):
    raw_chain_feats, metadata = du.process_pdb_file_custom(
        pdb_path,
        [
            summary["heavy_chain"],
            summary["light_chain"],
            *sorted(summary["antigen_chains"]),
        ],
    )
    chain_feats = process_chain(raw_chain_feats)
    chain_feats["pdb_name"] = metadata["pdb_name"]
    return chain_feats


@torch.no_grad()
def run_tokenize(
    struct_tokenizer: VQModel, input_pdb_folder, output_dir, summary_path: str
):
    with open(summary_path, "r") as f:
        summary = json.load(f)
        label2summary = {
            "{}".format(
                chains2label(
                    item["pdb"],
                    item["heavy_chain"],
                    item["light_chain"],
                    sorted(item["antigen_chains"]),
                )
            ): item
            for item in summary
        }

    # 生成缓存文件路径（基于 summary_path 的哈希值）
    summary_hash = hashlib.md5(summary_path.encode()).hexdigest()[:8]
    cache_path = os.path.join(output_dir, f"tokenize_cache_{summary_hash}.pkl")
    print("Cache files saving to", cache_path)

    # 检查缓存是否存在
    all_data = []
    valid_idx = []
    if os.path.exists(cache_path):
        log.info(f"Loading cached tokenize results from {cache_path}")
        try:
            cached_data = du.read_pkl(
                cache_path, use_torch=True, map_location="cpu"
            )
            all_data = cached_data["all_data"]
            valid_idx = cached_data["valid_idx"]
            log.info(f"Loaded {len(all_data)} cached samples")
        except Exception as e:
            log.warning(f"Failed to load cache: {e}. Will regenerate cache.")
            all_data = []
            valid_idx = []

    # 如果缓存不存在或加载失败，重新处理
    if not all_data:
        pBar = tqdm(
            summary,
            ncols=100,
            desc="Tokenize Complex",
        )
        error_count = 0
        for idx, item in enumerate(pBar):
            label = "{}.pdb".format(
                chains2label(
                    item["pdb"],
                    item["heavy_chain"],
                    item["light_chain"],
                    sorted(item["antigen_chains"]),
                )
            )
            pdb_path = os.path.join(input_pdb_folder, label)

            pBar.set_description(f"Tokenize: {label}")
            try:
                feats = load_from_pdb(
                    pdb_path,
                    process_chain=struct_tokenizer.process_chain,
                    summary=item,
                )
            except Exception:
                error_count += 1
                pBar.set_postfix(error=f"Error: {error_count}/{len(pBar)}")
                continue

            feats["pdb_path"] = pdb_path
            feats["header"] = feats["pdb_name"]

            all_data.append(feats)
            valid_idx.append(idx)

        # 保存缓存
        os.makedirs(output_dir, exist_ok=True)
        cache_data = {
            "all_data": all_data,
            "valid_idx": valid_idx,
        }
        du.write_pkl(cache_path, cache_data, create_dir=True, use_torch=True)
        log.info(f"Saved tokenize cache to {cache_path}")

    final_dataset: list[dict] = []
    all_header_struct_seq = []
    all_header_aa_seq = []
    pBar = tqdm(
        DataLoader(
            all_data,  # type: ignore
            batch_size=1,
            shuffle=False,
            drop_last=False,
            # collate_fn=collate_fn,
        ),
        ncols=100,
    )
    device = next(struct_tokenizer.parameters()).device
    for idx, batch in enumerate(pBar):
        pdb_name = batch["pdb_name"][0]
        _summary = label2summary[pdb_name]
        pBar.set_description(
            f"Tokenize: {pdb_name} (L={batch['seq_length'][0]})"
        )
        batch = recursive_to(batch, device)

        # struct token
        struct_ids = struct_tokenizer.tokenize(
            batch["all_atom_positions"],  # type: ignore
            batch["res_mask"],  # type: ignore
            batch["seq_length"],  # type: ignore
        )
        struct_token_list = struct_tokenizer.struct_ids_to_seq(
            struct_ids.cpu().tolist()[0]
        ).split(",")
        all_header_struct_seq.append((pdb_name, struct_token_list))

        # seq token
        aa_seq = du.aatype_to_seq(batch["aatype"].cpu().tolist()[0])  # type: ignore
        all_header_aa_seq.append((pdb_name, aa_seq))

        # add to final dataset
        hBeg, hEnd = _summary["heavy_chain_pos"]
        lBeg, lEnd = _summary["light_chain_pos"]

        final_dataset.append(
            {
                # token
                "heavy_struct_tok_seq": ",".join(struct_token_list[hBeg:hEnd]),
                "light_struct_tok_seq": ",".join(struct_token_list[lBeg:lEnd]),
                "epitope_struct_tok_seq": ",".join(
                    [struct_token_list[pos] for _, pos in _summary["epitope"]]
                ),
                "heavy_aa_tok_seq": aa_seq[hBeg:hEnd],
                "light_aa_tok_seq": aa_seq[lBeg:lEnd],
                "epitope_aa_tok_seq": "".join(
                    [aa_seq[pos] for _, pos in _summary["epitope"]]
                ),
                "pdb_name": pdb_name,
                # metadata
                "cdrh1_pos": _summary["cdrh1_pos"],
                "cdrh2_pos": _summary["cdrh2_pos"],
                "cdrh3_pos": _summary["cdrh3_pos"],
                "cdrl1_pos": _summary["cdrl1_pos"],
                "cdrl2_pos": _summary["cdrl2_pos"],
                "cdrl3_pos": _summary["cdrl3_pos"],
                "light_chain_pos": _summary["light_chain_pos"],
                "heavy_chain_pos": _summary["heavy_chain_pos"],
                "antigen_chain_pos": _summary["antigen_chain_pos"],
                "epitope": [pos for res, pos in _summary["epitope"]],
            }
        )

    dataset = Dataset.from_list(final_dataset)
    output_parquet_path = os.path.join(output_dir, "tokenized_complex.parquet")
    dataset.to_parquet(output_parquet_path)
    log.info(f"Saved {len(final_dataset)} samples to {output_parquet_path}")

    output_struct_fasta_path = os.path.join(output_dir, "struct_seq.fasta")
    fasta.FastaFile.write_iter(output_struct_fasta_path, all_header_struct_seq)

    output_aa_fasta_path = os.path.join(output_dir, "aa_seq.fasta")
    fasta.FastaFile.write_iter(output_aa_fasta_path, all_header_aa_seq)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_pdb_dir",
        type=str,
        default="/path/to/input/pdb/folder",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./generation-results/tokenized_protein",
    )
    parser.add_argument(
        "--summary_path",
        type=str,
    )
    parser.add_argument(
        "--struct_tokenizer_path",
        type=str,
        default="airkingbd/struct_tokenizer",
    )
    args = parser.parse_args()

    struct_tokenizer = get_struct_tokenizer(args.struct_tokenizer_path)
    struct_tokenizer = struct_tokenizer.cuda()
    run_tokenize(
        struct_tokenizer,
        args.input_pdb_dir,
        args.output_dir,
        args.summary_path,
    )


if __name__ == "__main__":
    main()

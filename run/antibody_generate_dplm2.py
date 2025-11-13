import argparse
import json
import os
from dataclasses import dataclass
from pprint import pprint

import biotite.sequence.io.fasta as fasta
import numpy as np
import torch
from Bio.PDB import PDBParser
from peft.peft_model import PeftModel

from byprot.datamodules.dataset.tokenized_protein import DPLM2Tokenizer
from byprot.datamodules.pdb_dataset import protein
from byprot.models.dplm2.dplm2 import (
    MultimodalDiffusionProteinLanguageModel as DPLM2,
)
from byprot.models.structok.structok_lfq import VQModel
from generate_dplm2 import save_fasta


# region Monkey patching
# Add chain separation methods to VQModel
def output_to_pdb_with_chain_separation(
    self,
    decoder_out,
    output_dir,
    chain_lengths,
    chain_names,
    original_pdb_path=None,
):
    """
    Convert decoder output to PDB files with proper chain separation.
    All chains are saved in a single PDB file with correct chain IDs.

    Args:
        decoder_out: Decoder output dictionary
        output_dir: Output directory for PDB files
        chain_lengths: List of lengths for each chain [light_chain_len, heavy_chain_len, antigen_chain_len]
        chain_names: List of chain names [light_chain_name, heavy_chain_name, antigen_chain_name]
        original_pdb_path: Path to original IMGT-encoded PDB file for residue indices
    """
    decoder_out = {kk: vv for kk, vv in decoder_out.items() if not kk == "sm"}
    headers = decoder_out.pop("header")

    # Generate PDB strings for all samples at once
    pdb_strings = self.decoder.output_to_pdb(decoder_out)

    # Process each sample
    for header, pdb_string in zip(headers, pdb_strings):
        separated_prot = _create_separated_protein(
            protein.from_pdb_string(pdb_string),
            chain_lengths,
            chain_names,
            original_pdb_path,
        )

        with open(os.path.join(output_dir, f"{header}.pdb"), "w") as f:
            pdb_prot = protein.to_pdb_with_custom_chains(  # type: ignore
                separated_prot, add_end=True, custom_chain_names=chain_names
            )
            f.write(pdb_prot)


def _create_separated_protein(
    prot, chain_lengths, origin_chain_names, original_pdb_path
):
    """
    Create a protein with proper chain separation from concatenated structure.

    Args:
        prot: Protein object containing concatenated structure
        chain_lengths: List of lengths for each chain
        chain_names: List of chain names for each chain
        original_pdb_path: Path to original IMGT-encoded PDB file for residue indices

    Returns:
        Protein object with proper chain separation
    """
    # Calculate total length
    total_length = sum(chain_lengths)

    # Initialize arrays for the separated protein
    separated_atom_positions = np.zeros((total_length, 37, 3))
    separated_atom_mask = np.zeros((total_length, 37))
    separated_aatype = np.zeros(total_length, dtype=np.int64)
    separated_residue_index = np.zeros(total_length, dtype=np.int64)
    separated_chain_index = np.zeros(total_length, dtype=np.int64)
    separated_b_factors = np.zeros((total_length, 37))

    # Load original residue indices if provided
    parser = PDBParser(QUIET=True)
    origin_chain_names = [
        chain.id
        for chain in parser.get_structure(
            "complex", original_pdb_path
        ).get_chains()
    ]
    with open(original_pdb_path, "r") as f:
        original_pdb_content = f.read()
        original_prot = protein.from_pdb_string(original_pdb_content)
        original_chain_indices = original_prot.chain_index
        original_residue_indices = original_prot.residue_index
        chain_residue_indices = {}
        for chain_idx in np.unique(original_chain_indices):
            chain_mask = original_chain_indices == chain_idx
            chain_residue_indices[origin_chain_names[chain_idx]] = (
                original_residue_indices[chain_mask]
            )

    start_idx = 0
    for chain_idx, (chain_len, chain_name) in enumerate(
        zip(chain_lengths, origin_chain_names)
    ):
        end_idx = start_idx + chain_len

        chain_atom_positions = prot.atom_positions[start_idx:end_idx]
        chain_atom_mask = prot.atom_mask[start_idx:end_idx]
        chain_aatype = prot.aatype[start_idx:end_idx]
        chain_b_factors = prot.b_factors[start_idx:end_idx]

        origin_chain_name = None
        for orig_chain_idx, orig_residues in chain_residue_indices.items():
            if len(orig_residues) == chain_len:
                origin_chain_name = orig_chain_idx
                chain_residue_index = chain_residue_indices[origin_chain_name]
                chain_index = np.full(chain_len, chain_idx, dtype=np.int64)
                break
        else:
            raise ValueError(
                f"No original chain found within {chain_residue_indices.keys()}."
            )

        # Assign to the separated protein arrays
        separated_atom_positions[start_idx:end_idx] = chain_atom_positions
        separated_atom_mask[start_idx:end_idx] = chain_atom_mask
        separated_aatype[start_idx:end_idx] = chain_aatype
        separated_residue_index[start_idx:end_idx] = chain_residue_index
        separated_chain_index[start_idx:end_idx] = chain_index
        separated_b_factors[start_idx:end_idx] = chain_b_factors

        start_idx = end_idx

    # Create the separated protein object
    separated_prot = protein.Protein(
        atom_positions=separated_atom_positions,
        atom_mask=separated_atom_mask,
        aatype=separated_aatype,
        residue_index=separated_residue_index,
        chain_index=separated_chain_index,
        b_factors=separated_b_factors,
    )

    return separated_prot


def _split_structure_by_chains(prot, chain_lengths, chain_names):
    """
    Split a concatenated protein structure into separate chains.

    Args:
        prot: Protein object containing concatenated structure
        chain_lengths: List of lengths for each chain
        chain_names: List of chain names for each chain

    Returns:
        List of Protein objects, one for each chain
    """
    chain_structures = []
    start_idx = 0

    for chain_idx, (chain_len, chain_name) in enumerate(
        zip(chain_lengths, chain_names)
    ):
        end_idx = start_idx + chain_len

        # Extract data for this chain
        chain_atom_positions = prot.atom_positions[start_idx:end_idx]
        chain_atom_mask = prot.atom_mask[start_idx:end_idx]
        chain_aatype = prot.aatype[start_idx:end_idx]
        chain_residue_index = prot.residue_index[start_idx:end_idx]
        chain_b_factors = prot.b_factors[start_idx:end_idx]

        # Create new residue indices starting from 1 for each chain
        chain_residue_index = np.arange(1, chain_len + 1)

        # Create chain index (all residues in this chain have the same chain index)
        chain_index = np.full(chain_len, chain_idx, dtype=np.int64)

        # Create Protein object for this chain
        chain_prot = protein.Protein(
            atom_positions=chain_atom_positions,
            atom_mask=chain_atom_mask,
            aatype=chain_aatype,
            residue_index=chain_residue_index,
            chain_index=chain_index,
            b_factors=chain_b_factors,
        )

        chain_structures.append(chain_prot)
        start_idx = end_idx

    return chain_structures


VQModel.output_to_pdb_with_chain_separation = (  # type: ignore
    output_to_pdb_with_chain_separation
)
VQModel._create_separated_protein = _create_separated_protein  # type: ignore
VQModel._split_structure_by_chains = _split_structure_by_chains  # type: ignore


# Monkey patch protein.to_pdb to support custom chain names
def to_pdb_with_custom_chains(
    prot: protein.Protein, model=1, add_end=True, custom_chain_names=None
) -> str:
    """
    Converts a `Protein` instance to a PDB string with custom chain names.

    Args:
        prot: The protein to convert to PDB.
        model: Model number for PDB format.
        add_end: Whether to add END record.
        custom_chain_names: List of custom chain names to use instead of default ABCD...

    Returns:
        PDB string with custom chain names.
    """
    from byprot.datamodules.pdb_dataset import residue_constants

    restypes = residue_constants.restypes + ["X"]

    def res_1to3(r):
        return residue_constants.restype_1to3.get(restypes[r], "UNK")

    atom_types = residue_constants.atom_types

    pdb_lines = []

    atom_mask = prot.atom_mask
    aatype = prot.aatype
    atom_positions = prot.atom_positions
    residue_index = prot.residue_index.astype(int)
    chain_index = prot.chain_index.astype(int)
    b_factors = prot.b_factors

    if np.any(aatype > residue_constants.restype_num):
        raise ValueError("Invalid aatypes.")

    # Use custom chain names if provided, otherwise use default ABCD...
    if custom_chain_names is not None:
        if len(custom_chain_names) < len(np.unique(chain_index)):
            raise ValueError(
                f"Not enough custom chain names provided. Need {len(np.unique(chain_index))}, got {len(custom_chain_names)}"
            )
        chain_ids = {i: custom_chain_names[i] for i in np.unique(chain_index)}
    else:
        # Use default chain ID mapping
        chain_ids = {}
        for i in np.unique(chain_index):
            if i >= protein.PDB_MAX_CHAINS:
                raise ValueError(
                    f"The PDB format supports at most {protein.PDB_MAX_CHAINS} chains."
                )
            chain_ids[i] = protein.PDB_CHAIN_IDS[i]

    pdb_lines.append(f"MODEL     {model}")
    atom_index = 1
    last_chain_index = chain_index[0]

    # Add all atom sites.
    for i in range(aatype.shape[0]):
        # Close the previous chain if in a multichain PDB.
        if last_chain_index != chain_index[i]:
            pdb_lines.append(
                protein._chain_end(
                    atom_index,
                    res_1to3(aatype[i - 1]),
                    chain_ids[chain_index[i - 1]],
                    residue_index[i - 1],
                )
            )
            last_chain_index = chain_index[i]
            atom_index += 1  # Atom index increases at the TER symbol.

        res_name_3 = res_1to3(aatype[i])
        for atom_name, pos, mask, b_factor in zip(
            atom_types, atom_positions[i], atom_mask[i], b_factors[i]
        ):
            if mask < 0.5:
                continue

            record_type = "ATOM"
            name = atom_name if len(atom_name) == 4 else f" {atom_name}"
            alt_loc = ""
            insertion_code = ""
            occupancy = 1.00
            element = atom_name[
                0
            ]  # Protein supports only C, N, O, S, this works.
            charge = ""
            # PDB is a columnar format, every space matters here!
            atom_line = (
                f"{record_type:<6}{atom_index:>5} {name:<4}{alt_loc:>1}"
                f"{res_name_3:>3} {chain_ids[chain_index[i]]:>1}"
                f"{residue_index[i]:>4}{insertion_code:>1}   "
                f"{pos[0]:>8.3f}{pos[1]:>8.3f}{pos[2]:>8.3f}"
                f"{occupancy:>6.2f}{b_factor:>6.2f}          "
                f"{element:>2}{charge:>2}"
            )
            pdb_lines.append(atom_line)
            atom_index += 1

    # Close the final chain.
    pdb_lines.append(
        protein._chain_end(
            atom_index,
            res_1to3(aatype[-1]),
            chain_ids[chain_index[-1]],
            residue_index[-1],
        )
    )
    pdb_lines.append("ENDMDL")
    if add_end:
        pdb_lines.append("END")

    # Pad all lines to 80 characters.
    pdb_lines = [line.ljust(80) for line in pdb_lines]
    return "\n".join(pdb_lines) + "\n"  # Add terminating newline.


# Apply monkey patching to protein module
protein.to_pdb_with_custom_chains = to_pdb_with_custom_chains  # type: ignore
# endregion Monkey patching


@dataclass
class MotifScaffoldingArgument:
    """DataClass for motif scaffolding arguments"""

    seed: int
    num_seqs: int
    experiment_path: str
    saveto: str
    temperature: float
    sampling_strategy: str
    max_iter: int
    batch_size: int
    dplm2_name_or_path: str
    struct_tokenizer_name_or_path: str
    seq_fasta: str
    struct_fasta: str
    summary: str
    target_cdr_types: list[str]


# region Prepare Inputs
def recursive_to(obj, device):
    if isinstance(obj, torch.Tensor):
        if device == "cpu":
            return obj.cpu()
        try:
            return obj.cuda(device=device, non_blocking=True)
        except RuntimeError:
            return obj.to(device)
    elif isinstance(obj, list):
        return [recursive_to(o, device=device) for o in obj]
    elif isinstance(obj, tuple):
        return tuple(recursive_to(o, device=device) for o in obj)
    elif isinstance(obj, dict):
        return {k: recursive_to(v, device=device) for k, v in obj.items()}

    else:
        return obj


def collate(
    tokenizer: DPLM2Tokenizer,
    init_aa_seq: list[str],
    init_struct_seq: list[str],
    device: str,
):
    batch_seq = tokenizer.batch_encode_plus(
        init_aa_seq,
        add_special_tokens=False,
        padding="longest",
        return_tensors="pt",
    )
    batch_seq = {
        "aa_ids": batch_seq["input_ids"],
        "aa_mask": batch_seq["attention_mask"].bool(),  # type: ignore
        "aa_targets": batch_seq["input_ids"].clone(),  # type: ignore
    }

    batch_struct = tokenizer.batch_encode_plus(
        init_struct_seq,
        add_special_tokens=False,
        padding="longest",
        return_tensors="pt",
    )
    batch_struct = {
        "struct_ids": batch_struct["input_ids"],
        "struct_mask": batch_struct["attention_mask"].bool(),  # type: ignore
        "struct_targets": batch_struct["input_ids"].clone(),  # type: ignore
    }
    batch = {
        "input_ids": torch.cat(
            (batch_struct["struct_ids"], batch_seq["aa_ids"]), dim=-1
        ),
        "input_mask": torch.cat(
            (batch_struct["struct_mask"], batch_seq["aa_mask"]), dim=-1
        ),
        "targets": torch.cat(
            (batch_struct["struct_targets"], batch_seq["aa_targets"]), dim=-1
        ),
    }
    batch.update(batch_struct)
    batch.update(batch_seq)

    # 0 stands for struct, 1 stands for aa, 2 stands for padding
    batch["type_ids"] = ((batch["input_ids"] < 33) & batch["input_mask"]).int()
    batch["type_ids"].masked_fill_(~batch["input_mask"], 2)
    batch = recursive_to(batch, device)

    # create partial mask
    seq_mask_idx = tokenizer.added_tokens_encoder[tokenizer.aa_mask_token]
    struct_mask_idx = tokenizer.added_tokens_encoder[
        tokenizer.struct_mask_token
    ]

    input_ids = batch["input_ids"]  # type: ignore
    input_mask = batch["input_mask"]  # type: ignore
    partial_mask = (
        input_ids.ne(seq_mask_idx)
        & input_ids.ne(struct_mask_idx)
        & input_ids.ne(tokenizer.pad_token_id)  # type: ignore
    ).type_as(input_mask)

    batch["partial_mask"] = partial_mask  # type: ignore

    return batch


def get_masked_seq_struct(
    sequence, structure, motif_pos_list, num_seqs, tokenizer
) -> tuple[list[str], list[str]]:
    seq_mask_token = tokenizer.aa_mask_token
    seq_bos_token = tokenizer.aa_cls_token
    seq_eos_token = tokenizer.aa_eos_token
    struct_mask_token = tokenizer.struct_mask_token
    struct_bos_token = tokenizer.struct_cls_token
    struct_eos_token = tokenizer.struct_eos_token

    masked_sequence = list(sequence)
    masked_structure = structure.split(",")
    for motif in motif_pos_list:
        beg, end = motif
        masked_sequence[beg : end + 1] = [seq_mask_token] * (end - beg + 1)
        masked_structure[beg : end + 1] = [struct_mask_token] * (end - beg + 1)

    masked_sequence = "".join([seq_bos_token, *masked_sequence, seq_eos_token])
    masked_structure = "".join(
        [struct_bos_token, *masked_structure, struct_eos_token]
    )

    return [masked_sequence for _ in range(num_seqs)], [
        masked_structure for _ in range(num_seqs)
    ]


def create_batches(batch, num_seqs, batch_size):
    batches = []
    start = 0
    end = start + batch_size
    while end < num_seqs:
        new_batch = {}
        for k, v in batch.items():
            new_batch[k] = v[start:end]
        batches.append(new_batch)
        start += batch_size
        end += batch_size
    assert end >= num_seqs
    # last batch if necessaryKJH
    if start < num_seqs:
        last_batch = {}
        for k, v in batch.items():
            last_batch[k] = v[start:end]
        batches.append(last_batch)
    return batches


def get_dplm2_batches(
    sequence,
    structure,
    motif_pos_list,
    args: MotifScaffoldingArgument,
    tokenizer,
    device,
):
    masked_sequence, masked_structure = get_masked_seq_struct(
        sequence, structure, motif_pos_list, args.num_seqs, tokenizer
    )

    batch = collate(tokenizer, masked_sequence, masked_structure, device)

    batches = create_batches(batch, args.num_seqs, args.batch_size)

    return batches


def summary2input(
    summary: dict, id2seq: dict, id2struct: dict, target_cdr_types: list[str]
) -> tuple[str, str, list[tuple[int, int]], dict]:
    light_seq = id2seq[summary["Label"] + "." + summary["Light-Chain"]]
    heavy_seq = id2seq[summary["Label"] + "." + summary["Heavy-Chain"]]
    light_struct = id2struct[summary["Label"] + "." + summary["Light-Chain"]]
    heavy_struct = id2struct[summary["Label"] + "." + summary["Heavy-Chain"]]
    antigen_seqs, antigen_structs = [], []
    for antigen_chain in summary["Antigen-Chains"]:
        antigen_seqs.append(id2seq[summary["Label"] + "." + antigen_chain])
        antigen_structs.append(
            id2struct[summary["Label"] + "." + antigen_chain]
        )

    sequence = light_seq + heavy_seq + "".join(antigen_seqs)
    structure = ",".join([light_struct, heavy_struct, "".join(antigen_structs)])

    cdr_pos_list: list = []
    for cdr_type in target_cdr_types:
        #  the position of CDR-H1/2/3 need to add the length of light chain
        cdr_name = "CDR-" + cdr_type
        if cdr_type.startswith("H"):
            cdr_pos_list.append(
                [
                    summary[cdr_name][0] + len(light_seq),
                    summary[cdr_name][1] + len(light_seq),
                ]
            )
        else:
            cdr_pos_list.append(summary[cdr_name])

    complex_info = {
        "label": summary["Label"],
        "pdb": summary["PDB-ID"],
        "pdb_path": summary["PDB-Path"],
        "light_seq": light_seq,
        "heavy_seq": heavy_seq,
        "antigen_seqs": antigen_seqs,
        "light_chain": summary["Light-Chain"],
        "heavy_chain": summary["Heavy-Chain"],
        "antigen_chains": summary["Antigen-Chains"],
        "cdr_types": target_cdr_types,
    }
    return sequence, structure, cdr_pos_list, complex_info


# endregion Prepare Inputs


# region Design
@torch.no_grad()
def motif_scaffolding(args: MotifScaffoldingArgument):
    # Loading model and tokenizer
    model: DPLM2 = DPLM2.from_pretrained(args.dplm2_name_or_path).eval().cuda()
    model.cfg.struct_tokenizer.exp_path = args.struct_tokenizer_name_or_path
    tokenizer: DPLM2Tokenizer = model.tokenizer
    device = next(model.parameters()).device
    if issubclass(type(model.net), PeftModel):
        model.net = model.net.merge_and_unload()

    # Read sequence and structure fasta files
    with (
        open(args.seq_fasta, "r") as f,
        open(args.struct_fasta, "r") as g,
        open(args.summary, "r") as h,
    ):
        id2seq = dict(fasta.FastaFile.read(f).items())
        id2struct = dict(fasta.FastaFile.read(g).items())
        summary = json.load(h)[:3]  # FIXME: Test for 5 samples

    for summary_item in summary:
        sequence, structure, motif_pos_list, complex_info = summary2input(
            summary_item, id2seq, id2struct, args.target_cdr_types
        )

        batches = get_dplm2_batches(
            sequence, structure, motif_pos_list, args, tokenizer, device
        )
        output_tokens = torch.tensor([], device=device)
        for batch in batches:
            with torch.cuda.amp.autocast():
                outputs = model.generate(
                    input_tokens=batch["input_ids"],
                    max_iter=args.max_iter,
                    sampling_strategy=args.sampling_strategy,
                    partial_masks=batch["partial_mask"],
                )["output_tokens"]
            output_tokens = torch.concat([output_tokens, outputs])  # pyright: ignore[reportArgumentType]

        # region saving
        # save scaffold fasta
        save_results(
            output_tokens=output_tokens,
            save_dir=os.path.join(args.saveto, summary_item["Label"]),
            tokenizer=tokenizer,
            struct_tokenizer=model.struct_tokenizer,  # type: ignore
            save_pdb=True,
            continue_write=True,
            complex_info=complex_info,
        )

        # endregion saving


# endregion Design


def save_results(
    tokenizer,
    struct_tokenizer: VQModel,
    save_dir,
    output_tokens,
    save_pdb,
    continue_write: bool,
    complex_info: dict,
):
    # save to fasta
    os.makedirs(save_dir, exist_ok=True)
    headers = [f"sample_{i}" for i in range(len(output_tokens))]

    struct_tokens, aatype_tokens = output_tokens.chunk(2, dim=-1)
    aatype_fasta_path = os.path.join(save_dir, "aatype.fasta")
    struct_tokens_strings = list(
        map(
            lambda s: ",".join(s.split()),
            tokenizer.batch_decode(struct_tokens, skip_special_tokens=True),
        )
    )
    aatype_strings = list(
        map(
            lambda s: "".join(s.split()),
            tokenizer.batch_decode(aatype_tokens, skip_special_tokens=True),
        )
    )
    save_fasta(
        save_name=aatype_fasta_path,
        output_results=aatype_strings,
        headers=headers,
        continue_write=continue_write,
    )

    prediction = []
    pdb_save_dir = os.path.join(save_dir, "structures")
    os.makedirs(pdb_save_dir, exist_ok=True)
    for header, aatype_str, struct_tokens_str in zip(
        headers, aatype_strings, struct_tokens_strings
    ):
        (
            aatype_tensor,
            struct_tokens_tensor,
        ) = struct_tokenizer.string_to_tensor(aatype_str, struct_tokens_str)
        decoder_out = struct_tokenizer.detokenize(struct_tokens_tensor)
        decoder_out["aatype"] = aatype_tensor
        decoder_out["header"] = [header]

        struct_tokenizer.output_to_pdb_with_chain_separation(
            decoder_out,
            output_dir=pdb_save_dir,
            chain_lengths=[
                len(complex_info["light_seq"]),
                len(complex_info["heavy_seq"]),
                *[len(seq) for seq in complex_info["antigen_seqs"]],
            ],
            chain_names=[
                complex_info["light_chain"],
                complex_info["heavy_chain"],
                *complex_info["antigen_chains"],
            ],
            original_pdb_path=complex_info["pdb_path"],
        )

        prediction.append(
            {
                "pdb": complex_info["label"],
                "heavy_chain": complex_info["heavy_chain"],
                "light_chain": complex_info["light_chain"],
                "antigen_chains": complex_info["antigen_chains"],
                "ref_pdb": complex_info["pdb_path"],
                "mod_pdb": os.path.abspath(
                    os.path.join(pdb_save_dir, header + ".pdb")
                ),
                "cdr_type": complex_info["cdr_types"],
                "struct_tokens": struct_tokens_str,
            }
        )

    with open(os.path.join(save_dir, "prediction.json"), "w") as f:
        json.dump(prediction, f, indent=2)

    return


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_seqs", type=int, default=20)
    parser.add_argument("--experiment_path", type=str)
    parser.add_argument("--saveto", type=str)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--sampling_strategy", type=str, default="annealing@2.0:1.0"
    )
    parser.add_argument("--max_iter", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--dplm2_name_or_path", type=str)
    parser.add_argument("--struct_tokenizer_name_or_path", type=str)
    parser.add_argument("--seq_fasta", type=str)
    parser.add_argument("--struct_fasta", type=str)
    parser.add_argument("--summary", type=str)
    parser.add_argument(
        "--target_cdr_types",
        nargs="+",
        default=["CDR-H3"],
        help="List of CDR types (e.g., --target_cdr_types CDR-H1 CDR-H2 CDR-H3)",
    )

    args = MotifScaffoldingArgument(**vars(parser.parse_args()))

    pprint(args)

    motif_scaffolding(args)


if __name__ == "__main__":
    main()

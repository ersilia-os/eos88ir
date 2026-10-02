"""Chemical Dice Integrator (CDI-Generalized): SMILES in, 8192-d embedding out.

CDI-Generalized (Kumar, Solanki et al., Nature Communications 2026,
https://doi.org/10.1038/s41467-026-77700-z) is IBM's SMI-SSED Mamba encoder finetuned to
predict the 8192-d CDI-Basic embedding, a fusion of six molecular views, from a SMILES
string alone. The model definition and the inference loop follow the authors' own server,
``api.py`` in https://huggingface.co/the-ahuja-lab/ChemicalDice (the GitHub copy under
``python-package/.../deployment/api.py`` builds a different class and does not match the
released checkpoint). ``SmiSsedPredictor`` below is copied from it unchanged.

What the wrapper adds, none of which touches the prediction path:

1. SMILES are canonicalised with RDKit before tokenising. The embedding identifies the
   SMILES *string*, not the molecule (a Kekule string retrieves its own canonical form
   only 12% of the time), and the authors' own client offers the same canonicalisation
   as ``convert_to_canonical``.
2. The checkpoint is the authors' ``model_state_dict`` alone. Their file also carries the
   AdamW state (790 of 1195 MB), which inference never reads. The IBM base checkpoint is
   not shipped: the finetuned weights overwrite every encoder tensor, and the only things
   taken from it are the 14 hyperparameters in ``smi_ssed_hparams.json`` (note IBM's
   checkpoint says ``bias=True``; the ``config.json`` in their repo says 0 and is wrong
   for this model) and the vocabulary.
3. The Mamba stack runs on CPU through ``mamba_cpu`` (see its docstring).
4. Weights are loaded with ``strict=True``; the authors' GitHub copy loads with
   ``strict=False``, which would let a key mismatch pass silently.
5. One molecule per forward pass, as in the authors' server. Their forward is not
   batch-safe (it takes ``outputs[0]`` of a ``[B, L, D]`` tensor), so batching would
   return wrong values for B > 1.

Behaviour inherited from the authors and left as is: the attention mask is ``ids != 0``
while the pad id is 2, so padding is not masked and every molecule is mean-pooled over the
same 127 positions; SMILES are cut to 128 tokens including <bos>/<eos>, so anything
longer than 126 tokens is silently truncated (a warning is printed here).
"""

import os
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch
from torch import nn
from rdkit import Chem, RDLogger

from smi_ssed import MolEncoder, MolTranBertTokenizer

RDLogger.DisableLog("rdApp.*")

EMB_DIM = 8192
MAX_LENGTH = 128

CODE_DIR = Path(__file__).resolve().parent
# model/checkpoints/ is gitignored: cdi_generalized.pt (405 MB) is hosted on eosvc and
# restored into this directory at pack time. Path stays relative to this file.
CHECKPOINT = Path(
    os.environ.get(
        "CDI_CHECKPOINT",
        CODE_DIR.parents[1] / "checkpoints" / "cdi_generalized.pt",
    )
)

_model = None
_tokenizer = None


class SmiSsedPredictor(nn.Module):
    """Verbatim from the authors' HF api.py."""

    def __init__(self, smi_ssed_model, embedding_dim):
        super().__init__()
        self.base_model_encoder = smi_ssed_model.encoder
        true_hidden_dim = self.base_model_encoder.mamba.embedding.weight.shape[1]
        self.regressor = nn.Linear(true_hidden_dim, embedding_dim)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model_encoder(input_ids, mask=attention_mask)
        hidden_states = outputs[0]

        if hidden_states.dim() == 2:
            batch_size = input_ids.shape[0]
            if batch_size > 0:
                seq_len = hidden_states.shape[0] // batch_size
                hidden_states = hidden_states.view(batch_size, seq_len, -1)

        expanded_mask = attention_mask.unsqueeze(-1)

        if expanded_mask.shape[1] != hidden_states.shape[1]:
            mask_for_pooling = torch.zeros_like(hidden_states)
            slice_len = min(expanded_mask.shape[1], hidden_states.shape[1])
            mask_for_pooling[:, :slice_len, :] = expanded_mask[:, :slice_len, :]
        else:
            mask_for_pooling = expanded_mask

        sum_hidden_states = torch.sum(hidden_states * mask_for_pooling, 1)
        sum_mask = torch.clamp(mask_for_pooling.sum(1), min=1e-9)
        molecular_representation = sum_hidden_states / sum_mask

        return self.regressor(molecular_representation)


def load_model():
    """Build tokenizer and model on CPU from the bundled files, once per process."""
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    if not CHECKPOINT.exists():
        raise FileNotFoundError("CDI checkpoint not found at %s" % CHECKPOINT)

    tokenizer = MolTranBertTokenizer(str(CODE_DIR / "bert_vocab_curated.txt"))
    with (CODE_DIR / "smi_ssed_hparams.json").open() as f:
        hparams = json.load(f)

    # The authors' class only needs an object exposing ``.encoder``.
    holder = types.SimpleNamespace(encoder=MolEncoder(hparams, len(tokenizer.vocab)))
    model = SmiSsedPredictor(holder, EMB_DIM)
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    # The thread count changes the last bits of the output (about 1e-6 between 1 and 8
    # threads), and scaling past one thread is poor for a 128-step scan on 1536-wide
    # tensors (1.2 s per molecule on one thread, 0.8 s on eight). One thread makes the
    # result independent of how many cores the host has or how busy it is.
    torch.set_num_threads(1)

    _model, _tokenizer = model, tokenizer
    return model, tokenizer


def canonicalise(smiles):
    """RDKit canonical (isomeric) SMILES, or None if the input is not a usable molecule.

    A non-string (an integer cell, or the NaN a reader yields for an empty cell) and the
    empty string are both bad rows, not reasons to stop the batch. RDKit parses ``""`` as
    a molecule with no atoms, which would otherwise be embedded as the bare <bos><eos>.
    """
    if not isinstance(smiles, str):
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


@torch.no_grad()
def embed_one(smiles, model, tokenizer):
    """The authors' per-molecule loop: tokenise, pad to 128, one forward pass."""
    enc = tokenizer.encode_plus(
        smiles,
        add_special_tokens=True,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].flatten().unsqueeze(0)
    attention_mask = (input_ids != 0).float()  # as in the authors' InferenceDataset
    return model(input_ids, attention_mask).squeeze(0).numpy()


def predict(smiles_list):
    """Embed a list of SMILES, returning an (n, 8192) float32 array.

    Molecules that cannot be parsed yield a row of NaN so the output stays aligned with
    the input, one row per molecule.
    """
    model, tokenizer = load_model()
    out = np.full((len(smiles_list), EMB_DIM), np.nan, dtype=np.float32)
    n_invalid = 0
    n_failed = 0
    n_truncated = 0
    for i, smiles in enumerate(smiles_list):
        # One bad row must never take down the batch, so any failure on a single
        # molecule (RDKit, tokenizer, forward pass) leaves that row as NaN.
        try:
            canonical = canonicalise(smiles)
            if canonical is None:
                n_invalid += 1
                continue
            # <bos> + tokens + <eos> must fit in MAX_LENGTH, else the tail is cut off.
            if len(tokenizer.regex_tokenizer.findall(canonical)) + 2 > MAX_LENGTH:
                n_truncated += 1
            out[i] = embed_one(canonical, model, tokenizer)
        except Exception:
            out[i] = np.nan
            n_failed += 1
    if n_invalid:
        print("CDI: %d input(s) could not be parsed as molecules; their rows are NaN" % n_invalid, file=sys.stderr)
    if n_failed:
        print("CDI: %d input(s) failed during processing; their rows are NaN" % n_failed, file=sys.stderr)
    if n_truncated:
        print(
            "CDI: %d SMILES have more than %d tokens and were truncated to their first %d "
            "(the %d-token limit includes <bos> and <eos>), as in the original model"
            % (n_truncated, MAX_LENGTH - 2, MAX_LENGTH - 2, MAX_LENGTH),
            file=sys.stderr,
        )
    return out

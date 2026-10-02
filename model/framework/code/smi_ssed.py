"""Tokenizer and encoder of IBM's SMILES state-space encoder decoder (SMI-SSED).

Source: https://github.com/IBM/materials (``smi_ssed/inference/smi_ssed/load.py``),
model card ``ibm-research/materials.smi-ssed``. Licensed under Apache-2.0 (see
THIRD_PARTY_NOTICES.md).

Only what the Chemical Dice model uses is kept: the regex tokenizer, the encoder
``MolEncoder`` and the ``LangLayer`` head (never called, but part of the checkpoint, so
it has to exist for the state dict to load with ``strict=True``). ``MolEncoder`` is
unchanged except that its Mamba stack comes from ``mamba_cpu`` instead of ``mamba_ssm``,
and ``LangLayer`` drops the ``.cuda()`` calls IBM makes inside ``forward``. The decoder,
the regression ``Net``, the training code and ``Smi_ssed.load_checkpoint`` are not used.
"""

import regex as re
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer

from mamba_cpu import MixerModel

# IBM writes this as a normal (non-raw) string, where ``\\\\`` is one literal backslash;
# the raw-string form below is the same regex.
PATTERN = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"


class MolTranBertTokenizer(BertTokenizer):
    def __init__(
        self,
        vocab_file: str = "",
        do_lower_case=False,
        unk_token="<pad>",
        sep_token="<eos>",
        pad_token="<pad>",
        cls_token="<bos>",
        mask_token="<mask>",
        **kwargs,
    ):
        super().__init__(
            vocab_file,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            **kwargs,
        )

        self.regex_tokenizer = re.compile(PATTERN)
        self.wordpiece_tokenizer = None
        self.basic_tokenizer = None
        with open(vocab_file) as f:
            self.padding_idx = f.readlines().index(pad_token + "\n")

    def _tokenize(self, text):
        split_tokens = self.regex_tokenizer.findall(text)
        return split_tokens


class LangLayer(nn.Module):
    def __init__(self, n_embd, n_vocab):
        super().__init__()
        self.embed = nn.Linear(n_embd, n_embd)
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, n_vocab, bias=False)


class MolEncoder(nn.Module):
    def __init__(self, config, n_vocab):
        super().__init__()

        self.config = config
        self.mamba = MixerModel(
            d_model=config["n_embd"],
            n_layer=config["n_layer"],
            ssm_cfg=dict(
                d_state=config["d_state"],
                d_conv=config["d_conv"],
                expand=config["expand_factor"],
                dt_rank=config["dt_rank"],
                dt_min=config["dt_min"],
                dt_max=config["dt_max"],
                dt_init=config["dt_init"],
                dt_scale=config["dt_scale"],
                dt_init_floor=config["dt_init_floor"],
                conv_bias=bool(config["conv_bias"]),
                bias=bool(config["bias"]),
            ),
            vocab_size=n_vocab,
        )

        # classification
        self.lang_model = LangLayer(config["n_embd"], n_vocab)

    def forward(self, idx, mask):
        x = self.mamba(idx)

        # add padding
        token_embeddings = x
        input_mask_expanded = mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        mask_embeddings = token_embeddings * input_mask_expanded
        token_embeddings = F.pad(
            mask_embeddings,
            pad=(0, 0, 0, self.config["max_len"] - mask_embeddings.shape[1]),
            value=0,
        )

        return token_embeddings

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from collections import OrderedDict

## Sequntial deep learning models

class GRUNet(nn.Module):
    """
    GRU-based model for predicting health events from longitudinal and current features.
    A GRU processes the longitudinal sequence, and its final hidden state is concatenated
    with the current features to make a prediction via a feedforward head.
    The output is a single logit for binary classification for each input in the batch.
    Longitudinal sequences are padded to the right, and a mask is provided to indicate valid steps.
    Args:
        nn (_type_): _description_
    """
    def __init__(
        self,
        input_size_curr: int,
        hidden_size_head: int,
        input_size_seq: int,
        hidden_size_seq: int,
        num_layers_seq: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_size_curr = input_size_curr
        self.hidden_size_head = hidden_size_head
        self.input_size_seq = input_size_seq
        self.hidden_size_seq = hidden_size_seq
        self.num_layers_seq = num_layers_seq

        # GRU over past sequence only
        self.gru = nn.GRU(
            input_size=input_size_seq,
            hidden_size=hidden_size_seq,
            num_layers=num_layers_seq,
            batch_first=True,
            dropout=0.0 if num_layers_seq == 1 else dropout,
        )

        # head over [summary_past || x_current]
        self.classifier_head = nn.Sequential(OrderedDict([
            ("fc1", nn.Linear(hidden_size_seq + input_size_curr, hidden_size_head)),
            ("relu1", nn.ReLU()),
            ("dropout1", nn.Dropout(dropout)),
            ("fc2", nn.Linear(hidden_size_head, 1)),
        ]))

    def has_attention(self) -> bool:
        return False
    
    def forward(self,
                x_current: torch.Tensor,
                x_past: torch.Tensor,
                mask_past: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x_past:    [B, T, D_long] float32  (past only; padded at the end)
            mask_past: [B, T]         bool     (True for valid steps in x_past)
            x_current: [B, D_curr]    float32  (current visit features)
        returns:   logits [B]
        """
        assert x_past is not None and mask_past is not None and x_current is not None, \
            "Must provide x_past, mask_past, and x_current"

        device = x_past.device
        B, T, D_long = x_past.shape

        # ensure boolean mask
        mask_past = mask_past.bool()
        lengths = mask_past.sum(dim=1)  # [B], number of valid past steps

        # placeholder summary
        h_last_past = torch.zeros(B, self.hidden_size_seq, device=device)

        has_past = lengths > 0
        if has_past.any():
            x_sel = x_past[has_past]                       # [B_sel, T, D_long]
            len_sel = lengths[has_past].to(torch.int64)    # [B_sel]

            # pack and run GRU
            packed = pack_padded_sequence(
                x_sel, lengths=len_sel.cpu(), batch_first=True, enforce_sorted=False
            )
            _, h_n = self.gru(packed)          # h_n: [num_layers, B_sel, H]
            h_last = h_n[-1]                   # [B_sel, H]
            h_last_past[has_past] = h_last


        feats = torch.cat([h_last_past, x_current], dim=1)   # [B, H + D_curr]
        logits = self.classifier_head(feats).squeeze(-1)  # [B]
        return logits

## Attention-pooling models

class AttentionPoolingNet(nn.Module):
    """
    Attention-pooling model for predicting health events from longitudinal and current features.
    Query is a learned vector

    Replaces the GRU with:
        h_k = W v_k
        s_k = a^T h_k + b
        alpha_k = softmax(s_k)  (masked over valid steps)
        z = sum_k alpha_k * h_k

    Then concatenates z with x_current and feeds through a small MLP head.
    The output is a single logit per example (binary classification).
    
    a: learned query vector for attention scoring.
    b: learned scalar bias for attention scoring.
    W: learned projection matrix for visit representations.
    s_k: attention score for visit k.
    alpha_k: attention weight for visit k.
    z: pooled representation of the longitudinal sequence.

    Args:
        input_size_curr:  D_curr, size of x_current.
        hidden_size_head: width of the hidden layer in the classifier head.
        input_size_seq:   D_long, size of each step in x_past.
        hidden_size_seq:  m, projection size used in attention (m < D_long is typical).
        num_layers_seq:   kept for interface compatibility; unused here.
        dropout:          dropout used in the classifier head.
    """
    def __init__(
        self,
        input_size_curr: int,
        hidden_size_head: int,
        input_size_seq: int,
        hidden_size_seq: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_size_curr = input_size_curr
        self.hidden_size_head = hidden_size_head
        self.input_size_seq = input_size_seq
        self.hidden_size_seq = hidden_size_seq
        self.droput = dropout

        # Linear projection: h_k = W v_k
        self.proj = nn.Linear(input_size_seq, hidden_size_seq, bias=False)

        # s_k = a^T h_k + b   (a in R^m, b scalar)
        self.attn_vec = nn.Parameter(torch.randn(hidden_size_seq) * 0.02)
        self.attn_bias = nn.Parameter(torch.zeros(()))

        # head over [z || x_current]
        self.classifier_head = nn.Sequential(OrderedDict([
            ("fc1", nn.Linear(hidden_size_seq + input_size_curr, hidden_size_head)),
            ("relu1", nn.ReLU()),
            ("dropout1", nn.Dropout(dropout)),
            ("fc2", nn.Linear(hidden_size_head, 1)),
        ]))

    def has_attention(self) -> bool:
        return True

    def _attend(self, H: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute masked attention weights over time steps.

        H:    [B_sel, T, m]
        mask: [B_sel, T]  (True for valid steps)
        returns alpha: [B_sel, T]
        """
        # scores s_k = a^T h_k + b
        s = H.matmul(self.attn_vec) + self.attn_bias
        s = s.masked_fill(~mask, float("-inf"))
        alpha = F.softmax(s, dim=1)
        # corner case: sample with no valid steps -> all -inf -> NaNs after softmax
        # we replace NaNs (if any) by zeros; caller will handle z=0 for those.
        alpha = torch.nan_to_num(alpha, nan=0.0)
        return alpha

    def forward(
        self,
        x_current: torch.Tensor,
        x_past: torch.Tensor,
        mask_past: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x_past:    [B, T, D_long]  (reverse chronological; padded at the END)
            mask_past: [B, T]          bool, True for valid steps
            x_current: [B, D_curr]
        returns:
            logits: [B]
        """
        assert x_past is not None and mask_past is not None and x_current is not None, \
            "Must provide x_past, mask_past, and x_current"

        device = x_past.device
        B, T, _ = x_past.shape
        mask_past = mask_past.bool()
        lengths = mask_past.sum(dim=1)  # [B]

        # default pooled representation for everyone
        z = torch.zeros(B, self.hidden_size_seq, device=device)
        alpha = None
        has_past = lengths > 0
        if has_past.any():
            x_sel = x_past[has_past]             # [B_sel, T, D_long]
            m_sel = mask_past[has_past]          # [B_sel, T]

            # project: H = W x
            # proj() applies linear layer to last dim
            H = self.proj(x_sel)                 # [B_sel, T, m]

            # attention weights (masked)
            alpha = self._attend(H, m_sel)       # [B_sel, T]

            # pooled: z = sum_t alpha_t * h_t
            # alpha.unsqueeze(1): [B_sel, 1, T]
            # bmm -> multiplies each matrix alpha [1, T] in the batch by H [T, m]
            z_sel = torch.bmm(alpha.unsqueeze(1), H).squeeze(1)  # [B_sel, m]
            z[has_past] = z_sel

        feats = torch.cat([z, x_current], dim=1)        # [B, m + D_curr]
        logits = self.classifier_head(feats).squeeze(-1)  # [B]
        return logits, alpha

class AttentionPoolingNetCurrentQuery(nn.Module):
    """
    Dot-product attention pooled over past visits, with a query from x_current.

    Keys:   k_t = W_k v_t
    Query:  q   = U_q x_current
    Scores: s_t = (q · k_t) / sqrt(d_k) + b
    Weights:alpha = softmax(masked s)
    Pooled: z = sum_t alpha_t * value_t   (value_t = k_t by default, or W_v v_t)

    Returns: logits [B], alpha [B, T]
    """
    def __init__(
        self,
        input_size_curr: int,      # D_curr
        hidden_size_head: int,
        input_size_seq: int,       # D_long
        hidden_size_seq: int,      # d_k (attention dimension)
        dropout: float = 0.0,
        use_separate_values: bool = False,  # if True, values = W_v v_t; else values = keys
        scale_scores: bool = True,          # divide by sqrt(d_k)
    ):
        super().__init__()
        self.input_size_curr = input_size_curr
        self.hidden_size_head = hidden_size_head
        self.input_size_seq = input_size_seq
        self.hidden_size_seq = hidden_size_seq
        self.use_separate_values = use_separate_values
        self.scale_scores = scale_scores

        d_k = hidden_size_seq

        # Key projection: k_t = W_k v_t
        self.key_proj = nn.Linear(input_size_seq, d_k, bias=False)

        # Optional distinct value projection: v^val_t = W_v v_t
        if use_separate_values:
            self.val_proj = nn.Linear(input_size_seq, d_k, bias=False)
        else:
            self.val_proj = None  # values = keys

        # Query from current features: q = U_q x_current
        self.query_proj = nn.Linear(input_size_curr, d_k, bias=False)

        # Optional scalar bias added to scores
        self.attn_bias = nn.Parameter(torch.zeros(()))

        # Head over [z || x_current]
        self.classifier_head = nn.Sequential(OrderedDict([
            ("fc1", nn.Linear(d_k + input_size_curr, hidden_size_head)),
            ("relu1", nn.ReLU()),
            ("dropout1", nn.Dropout(dropout)),
            ("fc2", nn.Linear(hidden_size_head, 1)),
        ]))

    def has_attention(self) -> bool:
        return True

    def _attend(
        self,
        K: torch.Tensor,          # [B_sel, T, d_k]  keys
        q: torch.Tensor,          # [B_sel, d_k]     query from x_current
        mask: torch.Tensor        # [B_sel, T]       True for valid steps
    ) -> torch.Tensor:
        """
        Compute masked, (optionally) scaled dot-product attention weights.
        Returns alpha: [B_sel, T]
        """
        d_k = K.size(-1)
        # scores s_t = (q · k_t) / sqrt(d_k) + b
        s = (K * q.unsqueeze(1)).sum(dim=-1)  # [B_sel, T]
        if self.scale_scores and d_k > 0:
            s = s / (d_k ** 0.5)
        s = s + self.attn_bias
        s = s.masked_fill(~mask, float("-inf"))
        alpha = F.softmax(s, dim=1)
        alpha = torch.nan_to_num(alpha, nan=0.0)  # handles all-pad rows safely
        return alpha

    def forward(
        self,
        x_current: torch.Tensor,     # [B, D_curr]
        x_past: torch.Tensor,        # [B, T, D_long] (padded at end)
        mask_past: torch.Tensor      # [B, T] (True for valid steps)
    ):
        assert x_past is not None and mask_past is not None and x_current is not None, \
            "Must provide x_past, mask_past, and x_current"

        device = x_past.device
        B, T, _ = x_past.shape
        mask = mask_past.bool()
        lengths = mask.sum(dim=1)  # [B]

        d_k = self.hidden_size_seq
        z = torch.zeros(B, d_k, device=device)
        alpha_full = torch.zeros(B, T, device=device)

        has_past = lengths > 0
        if has_past.any():
            x_sel = x_past[has_past]                 # [B_sel, T, D_long]
            m_sel = mask[has_past]                   # [B_sel, T]
            q_sel = self.query_proj(x_current[has_past])  # [B_sel, d_k]

            # Project to keys (and values)
            K = self.key_proj(x_sel)                 # [B_sel, T, d_k]
            V = self.val_proj(x_sel) if self.use_separate_values else K  # [B_sel, T, d_k]

            # Attention weights via helper
            alpha = self._attend(K, q_sel, m_sel)    # [B_sel, T]

            # Pool values with weights
            z_sel = torch.bmm(alpha.unsqueeze(1), V).squeeze(1)  # [B_sel, d_k]

            z[has_past] = z_sel
            alpha_full[has_past] = alpha

        feats = torch.cat([z, x_current], dim=1)          # [B, d_k + D_curr]
        logits = self.classifier_head(feats).squeeze(-1)   # [B]
        return logits, alpha_full

## Transformer-style cross-attention pooling

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.cos(pos * div)
        pe[:, 1::2] = torch.sin(pos * div)
        self.register_buffer("pe", pe)  # [max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0)

class CrossAttnPoolingNet(nn.Module):
    """
    Single-query cross-attention pooling over the longitudinal sequence, then MLP head.

    Replaces your scalar attention with:
        H = W * v_k  (+ positional enc)
        z = MHAttn(query = learned q, key=H, value=H, masked)
      where z is a single [B, m] vector.
    
    Keys (K): represent each visit → “what's available to attend to.”
    Query (Q): represents what kind of summary we want.
    Values (V): also derived from each visit → “what information we'll mix together.”
    
    z is a pooled summary for the whole sequence, a rich summary of the past visits.
    
    Cross-Attention is used, so the goal is to use a query vector out of the sequence 
    to attend over the projected visit representations H (both keys and values come from H).
    The attention score corresponds to how relevant each visit is to the learned query.

    Args:
        input_size_curr:  D_curr, size of x_current.
        hidden_size_head: width of the hidden layer in the classifier head.
        input_size_seq:   D_long, size of each step in x_past.
        hidden_size_seq:  m, attention model dimension (m < D_long typical).
        num_heads:        attention heads (e.g., 1-4).
        dropout:          dropout used in the classifier head.
        use_posenc:       add sinusoidal positional encodings to projected H.
    """
    def __init__(
        self,
        input_size_curr: int,
        hidden_size_head: int,
        input_size_seq: int,
        hidden_size_seq: int,
        num_heads: int = 2,
        dropout: float = 0.0,
        use_posenc: bool = True,
    ):
        super().__init__()
        assert hidden_size_seq % num_heads == 0, "hidden_size_seq must be divisible by num_heads"

        self.input_size_curr = input_size_curr
        self.hidden_size_head = hidden_size_head
        self.input_size_seq = input_size_seq
        self.hidden_size_seq = hidden_size_seq
        self.dropout = dropout  # (typo fixed vs. droput)

        # Project tokens: H_t = W v_t
        self.proj = nn.Linear(input_size_seq, hidden_size_seq, bias=False)

        # Optional positional encoding (makes attention order-aware)
        self.posenc = SinusoidalPositionalEncoding(hidden_size_seq) if use_posenc else nn.Identity()

        # Single learned query (one vector shared across the batch)
        self.query = nn.Parameter(torch.randn(1, 1, hidden_size_seq) * 0.02)

        # Multihead cross-attention: query -> (H, H)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size_seq,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )

        # Small MLP head on [z || x_current]
        self.classifier_head = nn.Sequential(OrderedDict([
            ("fc1", nn.Linear(hidden_size_seq + input_size_curr, hidden_size_head)),
            ("relu1", nn.ReLU()),
            ("dropout1", nn.Dropout(dropout)),
            ("fc2", nn.Linear(hidden_size_head, 1)),
        ]))

    def has_attention(self) -> bool:
        return True

    @torch.no_grad()
    def _make_kpmask(self, mask_valid: torch.Tensor) -> torch.Tensor:
        """
        Convert boolean 'valid' mask [B, T] (True=valid) into key_padding_mask [B, T] (True=PAD).
        """
        return ~mask_valid.bool()

    def forward(
        self,
        x_current: torch.Tensor,
        x_past: torch.Tensor,
        mask_past: torch.Tensor
    ):
        assert x_past is not None and mask_past is not None and x_current is not None

        B, T, _ = x_past.shape
        device = x_past.device

        # True where there is at least one valid step
        has_past = mask_past.bool().any(dim=1)        # [B]
        # Prepare outputs
        z = torch.zeros(B, self.hidden_size_seq, device=device)
        attn_weights_out = None

        if has_past.any():
            x_sel = x_past[has_past]                  # [B_sel, T, D]
            m_sel_valid = mask_past[has_past].bool()  # [B_sel, T]
            kpmask = ~m_sel_valid                     # True=PAD for MHA

            H = self.proj(x_sel)                      # [B_sel, T, m]
            H = self.posenc(H)                        # [B_sel, T, m]

            q = self.query.expand(H.size(0), 1, -1)   # [B_sel, 1, m]

            z_sel, attn_w = self.attn(
                query=q,
                key=H,
                value=H,
                key_padding_mask=kpmask,
                need_weights=True,
                average_attn_weights=False,
            )                                         # z_sel: [B_sel, 1, m], attn_w: [B_sel, h, 1, T]
            z_sel = z_sel.squeeze(1)                  # [B_sel, m]

            # Just in case: if any NaNs slipped through, zero them out
            z_sel = torch.nan_to_num(z_sel, nan=0.0, posinf=0.0, neginf=0.0)

            z[has_past] = z_sel

            # Average over heads to keep shape like your original attention map
            attn_weights_out = attn_w.mean(dim=1)     # [B_sel, 1, T]

        feats = torch.cat([z, x_current], dim=1)      # [B, m + D_curr]
        logits = self.classifier_head(feats).squeeze(-1)  # [B]

        # If you want a full [B, 1, T] attn map, stitch it back (zeros for empty)
        if attn_weights_out is not None:
            full_attn = torch.zeros(B, 1, T, device=device)
            full_attn[has_past] = attn_weights_out
        else:
            full_attn = None

        return logits, full_attn


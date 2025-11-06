"""
Conditional Counts Generator: Transformer Encoder with Cross-Attention
Predicts numerical counts (step durations) conditioned on symbolic sequences and context.
Non-autoregressive: all positions predicted in parallel.
Outputs: μ (mean) and σ (uncertainty) for each position.
"""
import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import COUNTS_MODEL_CONFIG, PAD_TOKEN_ID, VOCAB_SIZE
from models.layers import (
    PositionalEncoding, ConditioningProjection,
    CrossAttentionLayer, GammaOutputHead,
    create_key_padding_mask
)


class ConditionalCountsGenerator(nn.Module):
    """
    Transformer encoder with cross-attention for count prediction.

    Architecture:
        1. Conditioning encoder: processes patient/scan context
        2. Sequence encoder: processes symbolic sequence (sourceID + features)
        3. Cross-attention: attends from sequence to conditioning
        4. Dual output heads: predicts μ and σ for each position
    """

    def __init__(self, config=None):
        super(ConditionalCountsGenerator, self).__init__()

        if config is None:
            config = COUNTS_MODEL_CONFIG

        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.num_encoder_layers = config['num_encoder_layers']
        self.num_cross_attention_layers = config['num_cross_attention_layers']
        self.dim_feedforward = config['dim_feedforward']
        self.dropout = config['dropout']
        self.max_seq_len = config['max_seq_len']
        self.conditioning_dim = config['conditioning_dim']
        self.sequence_feature_dim = config['sequence_feature_dim']
        self.min_sigma = config.get('min_sigma', 0.1)

        # Conditioning projection
        self.conditioning_projection = ConditioningProjection(
            self.conditioning_dim, self.d_model, dropout=self.dropout
        )

        # Token embedding for symbolic sequence
        self.token_embedding = nn.Embedding(VOCAB_SIZE, self.d_model // 2, padding_idx=PAD_TOKEN_ID)

        # Projection for additional sequence features
        self.feature_projection = nn.Linear(2, self.d_model // 2)  # Position and Direction encoded

        # Combine token embeddings and features
        self.sequence_projection = nn.Linear(self.d_model, self.d_model)

        # Positional encoding
        self.pos_encoder_cond = PositionalEncoding(self.d_model, max_len=self.max_seq_len, dropout=self.dropout)
        self.pos_encoder_seq = PositionalEncoding(self.d_model, max_len=self.max_seq_len, dropout=self.dropout)

        # Conditioning encoder
        cond_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            batch_first=True
        )
        self.conditioning_encoder = nn.TransformerEncoder(
            cond_encoder_layer,
            num_layers=self.num_encoder_layers
        )

        # Sequence encoder
        seq_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            batch_first=True
        )
        self.sequence_encoder = nn.TransformerEncoder(
            seq_encoder_layer,
            num_layers=self.num_encoder_layers
        )

        # Cross-attention layers (sequence attends to conditioning)
        self.cross_attention_layers = nn.ModuleList([
            CrossAttentionLayer(
                d_model=self.d_model,
                nhead=self.nhead,
                dim_feedforward=self.dim_feedforward,
                dropout=self.dropout
            )
            for _ in range(self.num_cross_attention_layers)
        ])

        # Output head for Gamma distribution parameters
        self.output_head = GammaOutputHead(self.d_model, min_sigma=self.min_sigma)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode_conditioning(self, conditioning):
        """
        Encode conditioning features.

        Args:
            conditioning: [batch_size, conditioning_dim]

        Returns:
            cond_encoded: [batch_size, 1, d_model]
        """
        # Project conditioning
        cond_proj = self.conditioning_projection(conditioning)  # [batch_size, d_model]

        # Add sequence dimension
        cond_seq = cond_proj.unsqueeze(1)  # [batch_size, 1, d_model]

        # Add positional encoding
        cond_pos = self.pos_encoder_cond(cond_seq)

        # Encode
        cond_encoded = self.conditioning_encoder(cond_pos)

        return cond_encoded

    def encode_sequence(self, sequence_tokens, sequence_features, mask=None):
        """
        Encode symbolic sequence with additional features.

        Args:
            sequence_tokens: [batch_size, seq_len] - sourceID tokens
            sequence_features: [batch_size, seq_len, feature_dim] - Position, Direction
            mask: [batch_size, seq_len] - Boolean mask (True = valid)

        Returns:
            seq_encoded: [batch_size, seq_len, d_model]
        """
        # Embed tokens
        token_emb = self.token_embedding(sequence_tokens)  # [batch_size, seq_len, d_model/2]

        # Project features
        feature_emb = self.feature_projection(sequence_features)  # [batch_size, seq_len, d_model/2]

        # Concatenate
        combined = torch.cat([token_emb, feature_emb], dim=-1)  # [batch_size, seq_len, d_model]

        # Project to d_model
        seq_proj = self.sequence_projection(combined)

        # Add positional encoding
        seq_pos = self.pos_encoder_seq(seq_proj)

        # Create padding mask
        if mask is not None:
            key_padding_mask = ~mask  # Invert: True = padding
        else:
            key_padding_mask = create_key_padding_mask(sequence_tokens, pad_token_id=PAD_TOKEN_ID)

        # Encode
        seq_encoded = self.sequence_encoder(seq_pos, src_key_padding_mask=key_padding_mask)

        return seq_encoded

    def forward(self, conditioning, sequence_tokens, sequence_features, mask=None):
        """
        Forward pass to predict count parameters.

        Args:
            conditioning: [batch_size, conditioning_dim]
            sequence_tokens: [batch_size, seq_len]
            sequence_features: [batch_size, seq_len, feature_dim]
            mask: [batch_size, seq_len] - Boolean mask (True = valid)

        Returns:
            mu: [batch_size, seq_len] - mean parameters
            sigma: [batch_size, seq_len] - std deviation parameters
        """
        # Encode conditioning
        cond_encoded = self.encode_conditioning(conditioning)  # [batch_size, 1, d_model]

        # Encode sequence
        seq_encoded = self.encode_sequence(sequence_tokens, sequence_features, mask)  # [batch_size, seq_len, d_model]

        # Apply cross-attention layers
        cross_output = seq_encoded
        for cross_attn_layer in self.cross_attention_layers:
            cross_output = cross_attn_layer(
                query=cross_output,
                key_value=cond_encoded
            )

        # Predict μ and σ
        mu, sigma = self.output_head(cross_output)  # Each [batch_size, seq_len]

        return mu, sigma

    def compute_loss(self, mu, sigma, targets, mask=None):
        """
        Compute negative log-likelihood of Gamma distribution.

        Gamma parameterization:
            - shape (α) = (μ / σ)^2
            - rate (β) = μ / σ^2

        Args:
            mu: [batch_size, seq_len] - predicted means
            sigma: [batch_size, seq_len] - predicted standard deviations
            targets: [batch_size, seq_len] - true counts/durations
            mask: [batch_size, seq_len] - Boolean mask (True = valid)

        Returns:
            loss: Scalar loss value
        """
        # Compute Gamma parameters from μ and σ
        alpha = (mu / sigma) ** 2  # shape parameter
        beta = mu / (sigma ** 2)   # rate parameter

        # Ensure positive and numerical stability
        alpha = torch.clamp(alpha, min=1e-6)
        beta = torch.clamp(beta, min=1e-6)
        targets = torch.clamp(targets, min=1e-6)

        # Compute negative log-likelihood
        # log p(x|α,β) = (α-1)*log(x) - β*x + α*log(β) - log(Γ(α))
        log_prob = (
            (alpha - 1) * torch.log(targets)
            - beta * targets
            + alpha * torch.log(beta)
            - torch.lgamma(alpha)
        )

        # Apply mask if provided
        if mask is not None:
            log_prob = log_prob * mask.float()
            loss = -log_prob.sum() / mask.float().sum()
        else:
            loss = -log_prob.mean()

        return loss

    def sample_counts(self, mu, sigma, num_samples=1):
        """
        Sample counts from predicted Gamma distributions.

        Args:
            mu: [batch_size, seq_len] - predicted means
            sigma: [batch_size, seq_len] - predicted standard deviations
            num_samples: Number of samples to draw

        Returns:
            samples: [batch_size, seq_len, num_samples] - sampled counts
        """
        # Compute Gamma parameters
        alpha = (mu / sigma) ** 2
        beta = mu / (sigma ** 2)

        # Ensure positive
        alpha = torch.clamp(alpha, min=1e-6)
        beta = torch.clamp(beta, min=1e-6)

        # Sample from Gamma distribution
        # PyTorch uses (concentration=α, rate=β) parameterization
        gamma_dist = torch.distributions.Gamma(concentration=alpha, rate=beta)

        samples_list = []
        for _ in range(num_samples):
            sample = gamma_dist.sample()
            samples_list.append(sample)

        samples = torch.stack(samples_list, dim=-1)  # [batch_size, seq_len, num_samples]

        return samples


if __name__ == "__main__":
    # Test the model
    print("Testing Conditional Counts Generator...")

    # Create dummy config
    test_config = {
        'd_model': 128,
        'nhead': 4,
        'num_encoder_layers': 3,
        'num_cross_attention_layers': 2,
        'dim_feedforward': 512,
        'dropout': 0.1,
        'max_seq_len': 64,
        'conditioning_dim': 6,
        'sequence_feature_dim': 18,
        'min_sigma': 0.1
    }

    # Initialize model
    model = ConditionalCountsGenerator(test_config)
    print(f"\n✓ Model initialized")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create dummy inputs
    batch_size = 4
    seq_len = 20
    conditioning = torch.randn(batch_size, test_config['conditioning_dim'])
    sequence_tokens = torch.randint(1, 17, (batch_size, seq_len))
    sequence_features = torch.randn(batch_size, seq_len, 2)
    targets = torch.abs(torch.randn(batch_size, seq_len)) * 10  # Positive counts
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    mask[:, seq_len // 2:] = False  # Mask second half

    # Forward pass
    print(f"\n✓ Testing forward pass...")
    mu, sigma = model(conditioning, sequence_tokens, sequence_features, mask)
    print(f"  Conditioning shape: {conditioning.shape}")
    print(f"  Sequence tokens shape: {sequence_tokens.shape}")
    print(f"  Mu shape: {mu.shape}, range: [{mu.min():.2f}, {mu.max():.2f}]")
    print(f"  Sigma shape: {sigma.shape}, range: [{sigma.min():.2f}, {sigma.max():.2f}]")

    # Test loss
    loss = model.compute_loss(mu, sigma, targets, mask)
    print(f"\n✓ Loss computed: {loss.item():.4f}")

    # Test sampling
    print(f"\n✓ Testing sampling...")
    samples = model.sample_counts(mu, sigma, num_samples=3)
    print(f"  Samples shape: {samples.shape}")
    print(f"  Sample statistics: mean={samples.mean():.2f}, std={samples.std():.2f}")

    print("\n✓ All tests passed!")

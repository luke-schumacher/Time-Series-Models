"""
Temporal Schedule Model
Predicts daily MRI session structure (count, start times, gaps)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import TEMPORAL_MODEL_CONFIG


class TemporalScheduleModel(nn.Module):
    """
    Transformer-based model for predicting daily session structure

    Outputs:
        1. Session count (Poisson distribution)
        2. Session start times (Mixture of Gaussians)
    """

    def __init__(self, temporal_feature_dim=12, d_model=128, nhead=4, num_layers=4,
                 dim_feedforward=512, dropout=0.1, max_sessions=20, num_gaussian_components=3):
        """
        Initialize Temporal Schedule Model

        Args:
            temporal_feature_dim: Number of input temporal features
            d_model: Model embedding dimension
            nhead: Number of attention heads
            num_layers: Number of transformer encoder layers
            dim_feedforward: FFN hidden dimension
            dropout: Dropout probability
            max_sessions: Maximum sessions per day
            num_gaussian_components: Number of Gaussian components for start time mixture
        """
        super(TemporalScheduleModel, self).__init__()

        self.d_model = d_model
        self.max_sessions = max_sessions
        self.num_gaussian_components = num_gaussian_components

        # Feature projection
        self.feature_projection = nn.Sequential(
            nn.Linear(temporal_feature_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Session count head (Poisson distribution)
        self.session_count_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

        # Timing mixture head (Mixture of Gaussians)
        # Output: means, stds, and weights for K Gaussian components
        self.timing_mixture_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_gaussian_components * 3)  # K*(mu, sigma, weight)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, temporal_features):
        """
        Forward pass

        Args:
            temporal_features: [batch_size, temporal_feature_dim]

        Returns:
            session_count_lambda: Poisson lambda parameter [batch_size]
            timing_params: Dictionary with 'means', 'stds', 'weights' for mixture [batch_size, K]
        """
        batch_size = temporal_features.shape[0]

        # Project features
        features = self.feature_projection(temporal_features)  # [B, d_model]
        features = features.unsqueeze(1)  # [B, 1, d_model]

        # Encode through transformer
        encoded = self.transformer_encoder(features)  # [B, 1, d_model]
        encoded = encoded.squeeze(1)  # [B, d_model]

        # Session count prediction (Poisson lambda)
        session_count_logits = self.session_count_head(encoded).squeeze(-1)  # [B]
        session_count_lambda = F.softplus(session_count_logits) + 1.0  # Ensure lambda >= 1

        # Timing mixture parameters
        timing_logits = self.timing_mixture_head(encoded)  # [B, K*3]
        timing_logits = timing_logits.view(batch_size, self.num_gaussian_components, 3)

        # Extract means, stds, weights
        means = timing_logits[:, :, 0]  # [B, K] - seconds from midnight
        means = torch.sigmoid(means) * 14 * 3600  # Map to 0-14 hours (0-50400 seconds)

        stds = timing_logits[:, :, 1]  # [B, K]
        stds = F.softplus(stds) + 600  # Minimum std of 10 minutes (600 seconds)

        weights_logits = timing_logits[:, :, 2]  # [B, K]
        weights = F.softmax(weights_logits, dim=1)  # Normalize to sum to 1

        timing_params = {
            'means': means,
            'stds': stds,
            'weights': weights
        }

        return session_count_lambda, timing_params

    def predict_session_count(self, temporal_features):
        """
        Predict number of sessions

        Args:
            temporal_features: [batch_size, temporal_feature_dim]

        Returns:
            predicted_counts: [batch_size] - integer session counts
        """
        session_count_lambda, _ = self.forward(temporal_features)

        # Sample from Poisson distribution
        with torch.no_grad():
            poisson_dist = torch.distributions.Poisson(session_count_lambda)
            predicted_counts = poisson_dist.sample()
            predicted_counts = torch.clamp(predicted_counts, min=1, max=self.max_sessions)

        return predicted_counts.long()

    def sample_start_times(self, timing_params, num_sessions):
        """
        Sample session start times from mixture of Gaussians

        Args:
            timing_params: Dictionary with 'means', 'stds', 'weights' [1, K]
            num_sessions: Number of start times to sample

        Returns:
            start_times: [num_sessions] - sorted start times in seconds from midnight
        """
        means = timing_params['means'][0]  # [K]
        stds = timing_params['stds'][0]  # [K]
        weights = timing_params['weights'][0]  # [K]

        # Sample component indices
        component_dist = torch.distributions.Categorical(weights)
        component_indices = component_dist.sample((num_sessions,))  # [num_sessions]

        # Sample from selected Gaussians
        start_times = []
        for i in range(num_sessions):
            comp_idx = component_indices[i]
            mean = means[comp_idx]
            std = stds[comp_idx]

            # Sample from Gaussian
            normal_dist = torch.distributions.Normal(mean, std)
            start_time = normal_dist.sample()

            # Clamp to valid range (6 AM to 8 PM = 21600 to 72000 seconds)
            start_time = torch.clamp(start_time, min=21600, max=72000)
            start_times.append(start_time)

        # Sort start times
        start_times = torch.stack(start_times)
        start_times, _ = torch.sort(start_times)

        return start_times

    def predict_daily_structure(self, temporal_features, deterministic=False):
        """
        Predict complete daily structure

        Args:
            temporal_features: [1, temporal_feature_dim]
            deterministic: If True, use mean predictions instead of sampling

        Returns:
            daily_structure: Dictionary with:
                - 'num_sessions': int
                - 'session_start_times': list of floats (seconds from midnight)
                - 'session_gaps': list of floats (inter-session gaps in seconds)
        """
        session_count_lambda, timing_params = self.forward(temporal_features)

        # Predict session count
        if deterministic:
            num_sessions = int(torch.round(session_count_lambda[0]).item())
        else:
            num_sessions = self.predict_session_count(temporal_features)[0].item()

        num_sessions = max(1, min(num_sessions, self.max_sessions))

        # Sample start times
        start_times = self.sample_start_times(timing_params, num_sessions)
        start_times_list = start_times.cpu().numpy().tolist()

        # Calculate gaps
        if num_sessions > 1:
            gaps = np.diff(start_times_list).tolist()
        else:
            gaps = []

        return {
            'num_sessions': num_sessions,
            'session_start_times': start_times_list,
            'session_gaps': gaps
        }


def compute_poisson_nll_loss(predicted_lambda, target_counts):
    """
    Compute Poisson negative log-likelihood loss

    Args:
        predicted_lambda: Predicted Poisson lambda [batch_size]
        target_counts: Target session counts [batch_size]

    Returns:
        loss: Negative log-likelihood
    """
    # Poisson NLL: -log(P(k | lambda)) = lambda - k*log(lambda) + log(k!)
    # PyTorch provides Poisson NLL directly
    poisson_dist = torch.distributions.Poisson(predicted_lambda)
    nll = -poisson_dist.log_prob(target_counts)
    return nll.mean()


def compute_mixture_gaussian_nll_loss(timing_params, target_times):
    """
    Compute mixture of Gaussians NLL loss for start times

    Args:
        timing_params: Dictionary with 'means', 'stds', 'weights' [batch_size, K]
        target_times: Target start times [batch_size, max_sessions]

    Returns:
        loss: Negative log-likelihood
    """
    batch_size = timing_params['means'].shape[0]
    num_components = timing_params['means'].shape[1]

    means = timing_params['means']  # [B, K]
    stds = timing_params['stds']  # [B, K]
    weights = timing_params['weights']  # [B, K]

    # Compute log probabilities for each component
    # target_times: [B, N] where N is number of actual sessions (variable per batch)
    # We'll compute the mixture likelihood for each target time

    total_nll = 0
    count = 0

    for b in range(batch_size):
        target_times_b = target_times[b]  # [N]
        target_times_b = target_times_b[target_times_b >= 0]  # Filter out padding (-1)

        if len(target_times_b) == 0:
            continue

        for target_time in target_times_b:
            # Compute likelihood from each component
            component_likelihoods = []

            for k in range(num_components):
                normal_dist = torch.distributions.Normal(means[b, k], stds[b, k])
                log_prob = normal_dist.log_prob(target_time)
                weighted_log_prob = torch.log(weights[b, k] + 1e-8) + log_prob
                component_likelihoods.append(weighted_log_prob)

            # Log-sum-exp for numerical stability
            component_likelihoods = torch.stack(component_likelihoods)
            mixture_log_prob = torch.logsumexp(component_likelihoods, dim=0)

            total_nll -= mixture_log_prob
            count += 1

    return total_nll / max(count, 1)


def test_temporal_model():
    """Test the temporal schedule model"""
    print("Testing Temporal Schedule Model\n")

    # Create model
    model = TemporalScheduleModel(
        temporal_feature_dim=12,
        d_model=128,
        nhead=4,
        num_layers=4,
        max_sessions=20
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create dummy input
    batch_size = 4
    temporal_features = torch.randn(batch_size, 12)

    # Forward pass
    session_count_lambda, timing_params = model(temporal_features)

    print(f"\nSession count lambda: {session_count_lambda}")
    print(f"Timing means shape: {timing_params['means'].shape}")
    print(f"Timing stds shape: {timing_params['stds'].shape}")
    print(f"Timing weights shape: {timing_params['weights'].shape}")
    print(f"Timing weights (first sample): {timing_params['weights'][0]}")

    # Test prediction
    predicted_counts = model.predict_session_count(temporal_features)
    print(f"\nPredicted session counts: {predicted_counts}")

    # Test daily structure prediction
    daily_structure = model.predict_daily_structure(temporal_features[0:1])
    print(f"\nPredicted daily structure:")
    print(f"  Num sessions: {daily_structure['num_sessions']}")
    print(f"  Start times: {daily_structure['session_start_times'][:5]}...")
    print(f"  Gaps: {daily_structure['session_gaps'][:5]}...")

    # Test loss computation
    target_counts = torch.tensor([8, 12, 15, 10], dtype=torch.float32)
    count_loss = compute_poisson_nll_loss(session_count_lambda, target_counts)
    print(f"\nPoisson NLL loss: {count_loss.item():.4f}")

    print("\nAll tests passed!")


if __name__ == "__main__":
    test_temporal_model()

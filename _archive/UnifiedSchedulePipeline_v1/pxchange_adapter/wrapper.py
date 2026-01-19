"""
PXChange Model Adapter
Provides a unified interface to PXChange sequence and duration models
"""
import torch
import numpy as np
import pickle
import sys
import os

# Add PXChange to path
from config import PXCHANGE_DIR
sys.path.insert(0, PXCHANGE_DIR)

# Import PXChange models and utilities
from models.conditional_sequence_generator import ConditionalSequenceGenerator
from models.conditional_counts_generator import ConditionalCountsGenerator
import config as px_config


class PXChangeAdapter:
    """
    Adapter class for PXChange models
    Provides consistent interface for the unified pipeline
    """

    def __init__(self, sequence_model_path, duration_model_path, metadata_path, device='cpu'):
        """
        Initialize PXChange adapter

        Args:
            sequence_model_path: Path to trained sequence model checkpoint
            duration_model_path: Path to trained duration model checkpoint
            metadata_path: Path to preprocessing metadata (vocab, scalers, etc.)
            device: torch device ('cpu' or 'cuda')
        """
        self.device = torch.device(device)
        self.metadata_path = metadata_path

        # Load metadata
        self._load_metadata()

        # Initialize and load sequence model
        self.sequence_model = self._load_sequence_model(sequence_model_path)
        self.sequence_model.eval()

        # Initialize and load duration model
        self.duration_model = self._load_duration_model(duration_model_path)
        self.duration_model.eval()

        print(f"PXChange models loaded successfully on {device}")
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Conditioning features: {len(px_config.CONDITIONING_FEATURES)}")

    def _load_metadata(self):
        """Load preprocessing metadata"""
        with open(self.metadata_path, 'rb') as f:
            metadata = pickle.load(f)

        self.vocab = metadata.get('vocab', px_config.SOURCEID_VOCAB)
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)
        self.conditioning_scaler = metadata.get('conditioning_scaler', None)

        # Body group mappings
        self.bodygroup_map = metadata.get('bodygroup_map', {
            'HEAD': 0, 'NECK': 1, 'CHEST': 2, 'ABDOMEN': 3, 'PELVIS': 4,
            'SPINE': 5, 'ARM': 6, 'LEG': 7, 'HAND': 8, 'FOOT': 9, 'KNEE': 10
        })
        self.reverse_bodygroup_map = {v: k for k, v in self.bodygroup_map.items()}

    def _load_sequence_model(self, model_path):
        """Load sequence generator model"""
        model = ConditionalSequenceGenerator(
            vocab_size=self.vocab_size,
            d_model=px_config.SEQUENCE_MODEL_CONFIG['d_model'],
            nhead=px_config.SEQUENCE_MODEL_CONFIG['nhead'],
            num_encoder_layers=px_config.SEQUENCE_MODEL_CONFIG['num_encoder_layers'],
            num_decoder_layers=px_config.SEQUENCE_MODEL_CONFIG['num_decoder_layers'],
            dim_feedforward=px_config.SEQUENCE_MODEL_CONFIG['dim_feedforward'],
            dropout=px_config.SEQUENCE_MODEL_CONFIG['dropout'],
            max_seq_len=px_config.MAX_SEQ_LEN,
            conditioning_dim=len(px_config.CONDITIONING_FEATURES)
        ).to(self.device)

        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded sequence model from {model_path}")
        else:
            print(f"Warning: Sequence model not found at {model_path}. Using random initialization.")

        return model

    def _load_duration_model(self, model_path):
        """Load duration predictor model"""
        model = ConditionalCountsGenerator(
            d_model=px_config.COUNTS_MODEL_CONFIG['d_model'],
            nhead=px_config.COUNTS_MODEL_CONFIG['nhead'],
            num_encoder_layers=px_config.COUNTS_MODEL_CONFIG['num_encoder_layers'],
            num_cross_attention_layers=px_config.COUNTS_MODEL_CONFIG['num_cross_attention_layers'],
            dim_feedforward=px_config.COUNTS_MODEL_CONFIG['dim_feedforward'],
            dropout=px_config.COUNTS_MODEL_CONFIG['dropout'],
            max_seq_len=px_config.MAX_SEQ_LEN,
            conditioning_dim=len(px_config.CONDITIONING_FEATURES),
            sequence_feature_dim=self.vocab_size + len(px_config.SEQUENCE_FEATURE_COLUMNS),
            output_heads=px_config.COUNTS_MODEL_CONFIG['output_heads']
        ).to(self.device)

        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded duration model from {model_path}")
        else:
            print(f"Warning: Duration model not found at {model_path}. Using random initialization.")

        return model

    def prepare_conditioning(self, age, weight, height, bodygroup_from, bodygroup_to, ptab):
        """
        Prepare conditioning features

        Args:
            age: Patient age
            weight: Patient weight (kg)
            height: Patient height (m)
            bodygroup_from: Starting body group (string or int)
            bodygroup_to: Ending body group (string or int)
            ptab: Patient table position

        Returns:
            conditioning tensor [1, conditioning_dim]
        """
        # Convert body group strings to integers if needed
        if isinstance(bodygroup_from, str):
            bodygroup_from = self.bodygroup_map.get(bodygroup_from.upper(), 10)
        if isinstance(bodygroup_to, str):
            bodygroup_to = self.bodygroup_map.get(bodygroup_to.upper(), 10)

        conditioning = np.array([[age, weight, height, bodygroup_from, bodygroup_to, ptab]])

        # Scale if scaler is available
        if self.conditioning_scaler is not None:
            conditioning = self.conditioning_scaler.transform(conditioning)

        return torch.FloatTensor(conditioning).to(self.device)

    def generate_sequence(self, conditioning, max_length=None, temperature=1.0, top_k=10, top_p=0.9, seed=None):
        """
        Generate sourceID sequence

        Args:
            conditioning: Conditioning tensor [1, conditioning_dim]
            max_length: Maximum sequence length
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            seed: Random seed for reproducibility

        Returns:
            sequence_tokens: Generated token IDs [1, seq_len]
        """
        if seed is not None:
            torch.manual_seed(seed)

        if max_length is None:
            max_length = px_config.MAX_SEQ_LEN

        with torch.no_grad():
            sequence_tokens = self.sequence_model.generate(
                conditioning=conditioning,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )

        return sequence_tokens

    def predict_durations(self, conditioning, sequence_tokens):
        """
        Predict durations for a sequence

        Args:
            conditioning: Conditioning tensor [1, conditioning_dim]
            sequence_tokens: Sequence token IDs [1, seq_len]

        Returns:
            mu: Mean durations [1, seq_len]
            sigma: Std dev durations [1, seq_len]
            sampled_durations: Sampled durations from Gamma distribution [1, seq_len]
        """
        seq_len = sequence_tokens.shape[1]

        # Create sequence features (Position and Direction encoded)
        # For now, use zeros as placeholder - can be enhanced with actual features
        sequence_features = torch.zeros(1, seq_len, len(px_config.SEQUENCE_FEATURE_COLUMNS)).to(self.device)

        # Create mask (True for non-PAD tokens)
        mask = (sequence_tokens != px_config.PAD_TOKEN_ID)

        with torch.no_grad():
            mu, sigma = self.duration_model(conditioning, sequence_tokens, sequence_features, mask)
            sampled_durations = self.duration_model.sample_counts(mu, sigma, num_samples=1).squeeze(-1)

        return mu, sigma, sampled_durations

    def decode_tokens(self, token_ids):
        """
        Decode token IDs to sourceID strings

        Args:
            token_ids: Token IDs tensor or list

        Returns:
            sourceID strings list
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.cpu().numpy()

        if isinstance(token_ids, np.ndarray):
            if token_ids.ndim == 2:
                token_ids = token_ids[0]  # Take first batch item
            token_ids = token_ids.tolist()

        return [self.reverse_vocab.get(tid, 'UNK') for tid in token_ids]

    def decode_bodygroup(self, bodygroup_id):
        """Decode body group ID to string"""
        return self.reverse_bodygroup_map.get(int(bodygroup_id), 'Unknown')

    def generate_complete_session(self, age, weight, height, bodygroup_from, bodygroup_to, ptab,
                                    max_length=None, temperature=1.0, seed=None):
        """
        Generate complete session (sequence + durations)

        Args:
            age, weight, height, bodygroup_from, bodygroup_to, ptab: Conditioning features
            max_length: Maximum sequence length
            temperature: Sampling temperature
            seed: Random seed

        Returns:
            dict with 'tokens', 'sourceIDs', 'mu', 'sigma', 'durations'
        """
        # Prepare conditioning
        conditioning = self.prepare_conditioning(age, weight, height, bodygroup_from, bodygroup_to, ptab)

        # Generate sequence
        sequence_tokens = self.generate_sequence(conditioning, max_length, temperature, seed=seed)

        # Predict durations
        mu, sigma, durations = self.predict_durations(conditioning, sequence_tokens)

        # Decode tokens
        sourceIDs = self.decode_tokens(sequence_tokens)

        return {
            'tokens': sequence_tokens,
            'sourceIDs': sourceIDs,
            'mu': mu,
            'sigma': sigma,
            'durations': durations,
            'conditioning': {
                'age': age,
                'weight': weight,
                'height': height,
                'bodygroup_from': self.decode_bodygroup(bodygroup_from) if isinstance(bodygroup_from, (int, float)) else bodygroup_from,
                'bodygroup_to': self.decode_bodygroup(bodygroup_to) if isinstance(bodygroup_to, (int, float)) else bodygroup_to,
                'ptab': ptab
            }
        }


def test_adapter():
    """Test the PXChange adapter"""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from config import MODEL_PATHS

    adapter = PXChangeAdapter(
        sequence_model_path=MODEL_PATHS['pxchange']['sequence'],
        duration_model_path=MODEL_PATHS['pxchange']['duration'],
        metadata_path=MODEL_PATHS['pxchange']['metadata'],
        device='cpu'
    )

    # Generate a test session
    result = adapter.generate_complete_session(
        age=60,
        weight=80,
        height=1.75,
        bodygroup_from='HEAD',
        bodygroup_to='HEAD',
        ptab=-500000,
        seed=42
    )

    print("\nGenerated PXChange Session:")
    print(f"Sequence length: {len(result['sourceIDs'])}")
    print(f"SourceIDs: {result['sourceIDs'][:10]}...")
    print(f"Durations: {result['durations'][0, :10].tolist()}...")


if __name__ == "__main__":
    test_adapter()

"""
SeqofSeq Model Adapter
Provides a unified interface to SeqofSeq sequence and duration models
"""
import torch
import numpy as np
import pickle
import sys
import os

# Add SeqofSeq to path
from config import SEQOFSEQ_DIR
sys.path.insert(0, SEQOFSEQ_DIR)

# Import SeqofSeq models and utilities
from models.conditional_sequence_generator import ConditionalSequenceGenerator
from models.conditional_duration_predictor import ConditionalDurationPredictor
import config as seq_config


class SeqofSeqAdapter:
    """
    Adapter class for SeqofSeq models
    Provides consistent interface for the unified pipeline
    """

    def __init__(self, sequence_model_path, duration_model_path, metadata_path, device='cpu'):
        """
        Initialize SeqofSeq adapter

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

        print(f"SeqofSeq models loaded successfully on {device}")
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Conditioning features: {self.conditioning_dim}")

    def _load_metadata(self):
        """Load preprocessing metadata"""
        with open(self.metadata_path, 'rb') as f:
            metadata = pickle.load(f)

        self.vocab = metadata.get('vocab', {})
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)
        self.conditioning_scaler = metadata.get('conditioning_scaler', None)
        self.coil_columns = metadata.get('coil_columns', [])

        # Feature dimensions
        self.conditioning_dim = metadata.get('conditioning_dim', 92)  # 88 coils + 4 context features

        # Body part encoder
        self.bodypart_encoder = metadata.get('bodypart_encoder', None)

        # System type encoder
        self.systemtype_encoder = metadata.get('systemtype_encoder', None)

        # Country encoder
        self.country_encoder = metadata.get('country_encoder', None)

        # Group encoder
        self.group_encoder = metadata.get('group_encoder', None)

    def _load_sequence_model(self, model_path):
        """Load sequence generator model"""
        model = ConditionalSequenceGenerator(
            vocab_size=self.vocab_size,
            d_model=seq_config.SEQUENCE_MODEL_CONFIG['d_model'],
            nhead=seq_config.SEQUENCE_MODEL_CONFIG['nhead'],
            num_encoder_layers=seq_config.SEQUENCE_MODEL_CONFIG['num_encoder_layers'],
            num_decoder_layers=seq_config.SEQUENCE_MODEL_CONFIG['num_decoder_layers'],
            dim_feedforward=seq_config.SEQUENCE_MODEL_CONFIG['dim_feedforward'],
            dropout=seq_config.SEQUENCE_MODEL_CONFIG['dropout'],
            max_seq_len=seq_config.MAX_SEQ_LEN,
            conditioning_dim=self.conditioning_dim
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
        model = ConditionalDurationPredictor(
            d_model=seq_config.DURATION_MODEL_CONFIG['d_model'],
            nhead=seq_config.DURATION_MODEL_CONFIG['nhead'],
            num_encoder_layers=seq_config.DURATION_MODEL_CONFIG['num_encoder_layers'],
            num_cross_attention_layers=seq_config.DURATION_MODEL_CONFIG['num_cross_attention_layers'],
            dim_feedforward=seq_config.DURATION_MODEL_CONFIG['dim_feedforward'],
            dropout=seq_config.DURATION_MODEL_CONFIG['dropout'],
            max_seq_len=seq_config.MAX_SEQ_LEN,
            conditioning_dim=self.conditioning_dim,
            sequence_feature_dim=self.vocab_size + 2,  # vocab + 2 placeholder features
            output_heads=seq_config.DURATION_MODEL_CONFIG['output_heads'],
            min_sigma=seq_config.DURATION_TRAINING_CONFIG['min_sigma']
        ).to(self.device)

        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded duration model from {model_path}")
        else:
            print(f"Warning: Duration model not found at {model_path}. Using random initialization.")

        return model

    def prepare_conditioning(self, bodypart='HEAD', systemtype='VIDA', country='US', group='Brain', coil_config=None):
        """
        Prepare conditioning features

        Args:
            bodypart: Body part being scanned (string)
            systemtype: MRI system type (string)
            country: Country code (string)
            group: Scan group (string)
            coil_config: Dictionary or array of coil configurations

        Returns:
            conditioning tensor [1, conditioning_dim]
        """
        # Encode categorical features
        bodypart_enc = self.bodypart_encoder.transform([bodypart])[0] if self.bodypart_encoder else 0
        systemtype_enc = self.systemtype_encoder.transform([systemtype])[0] if self.systemtype_encoder else 0
        country_enc = self.country_encoder.transform([country])[0] if self.country_encoder else 0
        group_enc = self.group_encoder.transform([group])[0] if self.group_encoder else 0

        # Start with context features
        context_features = [bodypart_enc, systemtype_enc, country_enc, group_enc]

        # Add coil configuration (default to zeros if not provided)
        if coil_config is None:
            coil_features = [0] * len(self.coil_columns)
        elif isinstance(coil_config, dict):
            coil_features = [coil_config.get(col, 0) for col in self.coil_columns]
        else:
            coil_features = list(coil_config)

        # Combine all features
        conditioning = np.array([context_features + coil_features])

        # Scale if scaler is available
        if self.conditioning_scaler is not None:
            conditioning = self.conditioning_scaler.transform(conditioning)

        return torch.FloatTensor(conditioning).to(self.device)

    def generate_sequence(self, conditioning, max_length=None, temperature=1.0, top_k=10, top_p=0.9, seed=None):
        """
        Generate scan sequence

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
            max_length = seq_config.MAX_SEQ_LEN

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

        # Create sequence features (placeholder - can be enhanced)
        sequence_features = torch.zeros(1, seq_len, 2).to(self.device)

        # Create mask (True for non-PAD tokens)
        mask = (sequence_tokens != seq_config.PAD_TOKEN_ID)

        with torch.no_grad():
            mu, sigma = self.duration_model(conditioning, sequence_tokens, sequence_features, mask)
            sampled_durations = self.duration_model.sample_durations(mu, sigma, num_samples=1).squeeze(-1)

        return mu, sigma, sampled_durations

    def decode_tokens(self, token_ids):
        """
        Decode token IDs to scan sequence strings

        Args:
            token_ids: Token IDs tensor or list

        Returns:
            Scan sequence strings list
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.cpu().numpy()

        if isinstance(token_ids, np.ndarray):
            if token_ids.ndim == 2:
                token_ids = token_ids[0]  # Take first batch item
            token_ids = token_ids.tolist()

        return [self.reverse_vocab.get(tid, 'UNK') for tid in token_ids]

    def generate_complete_sequence(self, bodypart='HEAD', systemtype='VIDA', country='US', group='Brain',
                                     coil_config=None, max_length=None, temperature=1.0, seed=None):
        """
        Generate complete scan sequence (sequence + durations)

        Args:
            bodypart: Body part to scan
            systemtype: MRI system type
            country: Country code
            group: Scan group
            coil_config: Coil configuration
            max_length: Maximum sequence length
            temperature: Sampling temperature
            seed: Random seed

        Returns:
            dict with 'tokens', 'scan_sequences', 'mu', 'sigma', 'durations'
        """
        # Prepare conditioning
        conditioning = self.prepare_conditioning(bodypart, systemtype, country, group, coil_config)

        # Generate sequence
        sequence_tokens = self.generate_sequence(conditioning, max_length, temperature, seed=seed)

        # Predict durations
        mu, sigma, durations = self.predict_durations(conditioning, sequence_tokens)

        # Decode tokens
        scan_sequences = self.decode_tokens(sequence_tokens)

        return {
            'tokens': sequence_tokens,
            'scan_sequences': scan_sequences,
            'mu': mu,
            'sigma': sigma,
            'durations': durations,
            'conditioning': {
                'bodypart': bodypart,
                'systemtype': systemtype,
                'country': country,
                'group': group
            }
        }


def test_adapter():
    """Test the SeqofSeq adapter"""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from config import MODEL_PATHS

    adapter = SeqofSeqAdapter(
        sequence_model_path=MODEL_PATHS['seqofseq']['sequence'],
        duration_model_path=MODEL_PATHS['seqofseq']['duration'],
        metadata_path=MODEL_PATHS['seqofseq']['metadata'],
        device='cpu'
    )

    # Generate a test sequence
    result = adapter.generate_complete_sequence(
        bodypart='HEAD',
        systemtype='VIDA',
        country='KR',
        group='Brain',
        seed=42
    )

    print("\nGenerated SeqofSeq Sequence:")
    print(f"Sequence length: {len(result['scan_sequences'])}")
    print(f"Scan sequences: {result['scan_sequences'][:10]}...")
    print(f"Durations: {result['durations'][0, :10].tolist()}...")


if __name__ == "__main__":
    test_adapter()

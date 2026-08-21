"""Sampling protocol names for synthetic scans.

The protocol is a model INPUT read off a real MRI_MSR_100 message, and it is
the strongest one the examination duration model has: step 04's gate measures a
per-protocol group mean at held-out R2 76.3% / MAE 16.2s, against
sequence_type's 18.7% / 59.9s. A synthetic scan has no message, so generation
has to draw a protocol from the real empirical distribution — exactly what
`sut_parameter_sampling.SUTParameterSampler` already does for TR/num_slices,
and this module deliberately mirrors its shape.

WHY NOT JUST FEED THE RARE BUCKET. `sequence_generator` falls back to
RARE_PROTOCOL_ID whenever `body_region_info['protocol']` is absent, so doing
nothing is not neutral — it pins every synthetic scan to one embedding row that
was trained on the 3.4% of sequences whose protocol was too infrequent to earn
its own. The model loads cleanly and predicts a blend of leftovers.

WHY SERIAL COMES FIRST IN THE BACKOFF. Protocols are site-specific: ~324 per
serial, and only 4.1% of names appear on more than one scanner (see
`protocol_vocab`). A protocol drawn from the wrong site is a name that scanner
has never run, which breaks the one thing the synthetic Protocol column is for
— putting real and synthetic rows on the same Qlik dimension. So the first key
is (serial, region, type) and the fallbacks widen from there.

Both halves of the draw are returned. The ID is what the model conditions on;
the RAW NAME is what the synthetic CSV's Protocol column must carry, and they
have to come from the same draw or the two disagree about what was generated.
"""

from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

from .protocol_vocab import RARE_PROTOCOL_ID, normalize_protocol_name, protocol_id

MIN_POOL = 5  # below this a pool is too thin to be representative


class ProtocolSampler:
    """Draws (protocol_id, raw_name) pairs from the real training distribution."""

    def __init__(
        self,
        examination_sequences: Sequence[dict],
        vocab: Optional[Dict[str, int]] = None,
        rng=None,
    ):
        self.vocab = dict(vocab or {})
        self._rng = rng
        self._pools: Dict[tuple, List[Tuple[int, str]]] = defaultdict(list)

        for seq in examination_sequences or []:
            raw = seq.get('protocol_name')
            if not normalize_protocol_name(raw):
                # No name on this row: it carries no protocol information, so
                # letting it into a pool would make "unknown" a drawable
                # outcome at its corpus frequency rather than only where the
                # pools genuinely run out.
                continue

            # int(), not whatever the vocabulary JSON deserialised to:
            # SequenceGeneratorModel._ensure_batched promotes plain ints to
            # tensors via isinstance(val, int), which a numpy integer fails.
            entry = (int(protocol_id(raw, self.vocab)), str(raw))
            serial = int(seq.get('serial_idx', 0) or 0)
            region = int(seq.get('body_region', 0) or 0)
            seq_type = int(seq.get('sequence_type', 0) or 0)

            self._pools[('serial_region_type', serial, region, seq_type)].append(entry)
            self._pools[('region_type', region, seq_type)].append(entry)
            self._pools[('type', seq_type)].append(entry)
            self._pools[('global',)].append(entry)

        self._observations = len(self._pools.get(('global',), []))

    @property
    def observations(self) -> int:
        return self._observations

    def _random_index(self, size: int) -> int:
        if self._rng is not None:
            return int(self._rng.integers(size))
        import random
        return random.randrange(size)

    def sample(self, serial_idx, body_region, sequence_type) -> Tuple[int, str]:
        """Return one (protocol_id, raw_name) draw for this scan context."""
        keys = (
            ('serial_region_type', int(serial_idx), int(body_region), int(sequence_type)),
            ('region_type', int(body_region), int(sequence_type)),
            ('type', int(sequence_type)),
            ('global',),
        )
        for key in keys:
            pool = self._pools.get(key)
            if not pool:
                continue
            if len(pool) >= MIN_POOL or key == ('global',):
                return pool[self._random_index(len(pool))]

        # No training observations at all. RARE_PROTOCOL_ID is what the model
        # already substitutes for an absent key, so this is in distribution;
        # the empty name keeps the CSV honest about having invented nothing.
        return RARE_PROTOCOL_ID, ''

    def describe(self) -> str:
        contexts = sum(1 for k in self._pools if k[0] == 'serial_region_type')
        if not self._observations:
            return (
                "Protocol sampler: NO observations found — every synthetic scan will "
                "use RARE_PROTOCOL_ID, which pins the model's strongest duration "
                "input to one embedding row. Check that the pkl carries "
                "'protocol_name' on each sequence (step 03 writes it)."
            )
        distinct = len({entry[1] for entry in self._pools[('global',)]})
        rare = sum(1 for entry in self._pools[('global',)]
                   if entry[0] == RARE_PROTOCOL_ID)
        return (
            f"Protocol sampler: {self._observations:,} real observations of "
            f"{distinct:,} distinct protocol names across {contexts:,} "
            f"(serial, body_region, sequence_type) contexts. "
            f"{100 * rare / self._observations:.1f}% fall to the rare bucket "
            f"(below the vocabulary's min_count)."
        )


def build_protocol_sampler(
    preprocessed_data: dict,
    vocab: Optional[Dict[str, int]] = None,
    rng=None,
) -> Optional[ProtocolSampler]:
    """Build a sampler from a preprocessed pkl, or None when not applicable."""
    sequences = (preprocessed_data or {}).get('examination', [])
    if not sequences:
        return None
    return ProtocolSampler(sequences, vocab=vocab, rng=rng)

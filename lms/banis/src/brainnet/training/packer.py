"""
Sequence Packer for Efficient Training.

Packs multiple samples into fixed-length sequences (4K-16K tokens)
for efficient training on Phi-3.5.

Techniques:
- Greedy bin packing for optimal utilization
- Sample concatenation with separators
- Attention mask handling for packed sequences
- Truncation of oversized samples
"""

import logging
from typing import List, Dict, Any, Optional, Iterator, Tuple, Generator
from dataclasses import dataclass
import random

from .config import PackingConfig, TrainingConfig

logger = logging.getLogger(__name__)


try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers not installed. Run: pip install transformers")


@dataclass
class PackedSequence:
    """A packed sequence ready for training."""
    input_ids: List[int]
    attention_mask: List[int]
    labels: List[int]
    
    # Metadata
    num_samples: int
    sample_boundaries: List[Tuple[int, int]]  # (start, end) for each sample
    total_tokens: int
    padding_tokens: int
    
    @property
    def utilization(self) -> float:
        """Sequence utilization (non-padding ratio)."""
        if self.total_tokens == 0:
            return 0.0
        return (self.total_tokens - self.padding_tokens) / self.total_tokens


class SequencePacker:
    """
    Pack multiple samples into fixed-length sequences.
    
    Optimized for Phi-3.5 training with 4K-16K context windows.
    Uses greedy bin packing for high utilization (>95%).
    """
    
    def __init__(self, config: Optional[TrainingConfig] = None):
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError(
                "transformers required. Install with: pip install transformers"
            )
            
        self.config = config or TrainingConfig()
        self.packing_config = self.config.packing
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model,
            trust_remote_code=True,
        )
        
        # Ensure special tokens
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Get separator token IDs
        self.sep_ids = self.tokenizer.encode(
            self.packing_config.sep_token,
            add_special_tokens=False
        )
        
        # Stats
        self._total_samples = 0
        self._total_sequences = 0
        self._total_tokens = 0
        self._padding_tokens = 0
        
    def tokenize(self, text: str) -> List[int]:
        """Tokenize text without special tokens."""
        return self.tokenizer.encode(text, add_special_tokens=False)
    
    def _truncate_if_needed(self, token_ids: List[int]) -> List[int]:
        """Truncate sequence if it exceeds max length."""
        max_len = self.packing_config.max_seq_length - len(self.sep_ids) - 10
        
        if len(token_ids) > max_len:
            if self.packing_config.truncate_long_samples:
                return token_ids[:max_len]
            else:
                return None  # Skip sample
        return token_ids
    
    def pack_samples(
        self,
        samples: List[str],
        shuffle: bool = True,
    ) -> Generator[PackedSequence, None, None]:
        """
        Pack samples into sequences using greedy bin packing.
        
        Args:
            samples: List of text samples to pack
            shuffle: Shuffle samples before packing
            
        Yields:
            PackedSequence objects
        """
        # Tokenize all samples
        tokenized = []
        for sample in samples:
            ids = self.tokenize(sample)
            ids = self._truncate_if_needed(ids)
            if ids is not None:
                tokenized.append(ids)
                
        if shuffle:
            random.shuffle(tokenized)
            
        # Sort by length (descending) for better packing
        tokenized.sort(key=len, reverse=True)
        
        # Pack using first-fit decreasing
        yield from self._first_fit_packing(tokenized)
    
    def _first_fit_packing(
        self,
        tokenized_samples: List[List[int]]
    ) -> Generator[PackedSequence, None, None]:
        """
        First-fit decreasing bin packing algorithm.
        
        Achieves >95% utilization in practice.
        """
        target_len = self.packing_config.target_seq_length
        sep_len = len(self.sep_ids)
        
        # Bins: list of (current_length, [token_ids_list])
        bins: List[Tuple[int, List[List[int]]]] = []
        
        for sample_ids in tokenized_samples:
            sample_len = len(sample_ids) + sep_len
            
            # Try to fit in existing bin
            placed = False
            for i, (bin_len, bin_samples) in enumerate(bins):
                if bin_len + sample_len <= target_len:
                    bins[i] = (bin_len + sample_len, bin_samples + [sample_ids])
                    placed = True
                    break
                    
            # Create new bin if doesn't fit
            if not placed:
                bins.append((sample_len, [sample_ids]))
                
            # Yield completed bins
            while bins and bins[0][0] >= target_len * 0.9:
                bin_len, bin_samples = bins.pop(0)
                yield self._create_packed_sequence(bin_samples)
                
        # Yield remaining bins
        for bin_len, bin_samples in bins:
            if bin_samples:
                yield self._create_packed_sequence(bin_samples)
    
    def _create_packed_sequence(
        self,
        sample_ids_list: List[List[int]]
    ) -> PackedSequence:
        """Create a PackedSequence from list of tokenized samples."""
        target_len = self.packing_config.target_seq_length
        
        # Concatenate with separators
        combined_ids = []
        boundaries = []
        
        for sample_ids in sample_ids_list:
            start = len(combined_ids)
            combined_ids.extend(sample_ids)
            combined_ids.extend(self.sep_ids)
            end = len(combined_ids)
            boundaries.append((start, end))
            
        # Truncate if over target
        if len(combined_ids) > target_len:
            combined_ids = combined_ids[:target_len]
            
        # Pad if under target
        padding_needed = target_len - len(combined_ids)
        pad_id = self.tokenizer.pad_token_id
        
        input_ids = combined_ids + [pad_id] * padding_needed
        attention_mask = [1] * len(combined_ids) + [0] * padding_needed
        
        # Labels: same as input_ids, but with -100 for padding
        labels = combined_ids + [-100] * padding_needed
        
        # Update stats
        self._total_samples += len(sample_ids_list)
        self._total_sequences += 1
        self._total_tokens += target_len
        self._padding_tokens += padding_needed
        
        return PackedSequence(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            num_samples=len(sample_ids_list),
            sample_boundaries=boundaries,
            total_tokens=target_len,
            padding_tokens=padding_needed,
        )
    
    def pack_streaming(
        self,
        sample_iterator: Iterator[str],
        buffer_size: int = 1000,
    ) -> Generator[PackedSequence, None, None]:
        """
        Pack samples from a streaming iterator.
        
        Uses a buffer for efficient packing while maintaining
        memory efficiency for large datasets.
        
        Args:
            sample_iterator: Iterator yielding text samples
            buffer_size: Number of samples to buffer before packing
            
        Yields:
            PackedSequence objects
        """
        buffer = []
        
        for sample in sample_iterator:
            ids = self.tokenize(sample)
            ids = self._truncate_if_needed(ids)
            
            if ids is not None:
                buffer.append(ids)
                
            if len(buffer) >= buffer_size:
                # Sort and pack buffer
                buffer.sort(key=len, reverse=True)
                yield from self._first_fit_packing(buffer)
                buffer = []
                
        # Pack remaining samples
        if buffer:
            buffer.sort(key=len, reverse=True)
            yield from self._first_fit_packing(buffer)
    
    def pack_for_training(
        self,
        samples: List[str],
    ) -> List[Dict[str, List[int]]]:
        """
        Pack samples into training-ready format.
        
        Returns list of dicts with 'input_ids', 'attention_mask', 'labels'.
        Ready for HuggingFace Trainer.
        
        Args:
            samples: List of text samples
            
        Returns:
            List of training examples
        """
        result = []
        
        for packed in self.pack_samples(samples):
            result.append({
                'input_ids': packed.input_ids,
                'attention_mask': packed.attention_mask,
                'labels': packed.labels,
            })
            
        return result
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Get packing statistics."""
        utilization = 0.0
        if self._total_tokens > 0:
            utilization = (self._total_tokens - self._padding_tokens) / self._total_tokens
            
        return {
            'total_samples': self._total_samples,
            'total_sequences': self._total_sequences,
            'total_tokens': self._total_tokens,
            'padding_tokens': self._padding_tokens,
            'utilization': utilization,
            'avg_samples_per_sequence': (
                self._total_samples / self._total_sequences
                if self._total_sequences > 0 else 0
            ),
        }
    
    def reset_stats(self):
        """Reset statistics counters."""
        self._total_samples = 0
        self._total_sequences = 0
        self._total_tokens = 0
        self._padding_tokens = 0


class DynamicPacker:
    """
    Dynamic packer that adjusts sequence length based on content.
    
    For variable-length training (4K-16K tokens).
    """
    
    def __init__(
        self,
        config: Optional[TrainingConfig] = None,
        length_buckets: List[int] = None
    ):
        self.config = config or TrainingConfig()
        
        # Default buckets: 4K, 6K, 8K, 12K, 16K
        self.length_buckets = length_buckets or [4096, 6144, 8192, 12288, 16384]
        
        # Create packer for each bucket
        self.packers: Dict[int, SequencePacker] = {}
        
        for length in self.length_buckets:
            bucket_config = TrainingConfig()
            bucket_config.packing.target_seq_length = length
            bucket_config.packing.max_seq_length = length
            self.packers[length] = SequencePacker(bucket_config)
    
    def _select_bucket(self, token_count: int) -> int:
        """Select appropriate bucket for token count."""
        for bucket in self.length_buckets:
            if token_count <= bucket:
                return bucket
        return self.length_buckets[-1]
    
    def pack_with_dynamic_length(
        self,
        samples: List[str],
    ) -> Generator[PackedSequence, None, None]:
        """
        Pack samples using dynamic sequence lengths.
        
        Shorter samples go to shorter sequences for efficiency.
        """
        # Group samples by appropriate bucket
        buckets: Dict[int, List[str]] = {b: [] for b in self.length_buckets}
        
        ref_packer = self.packers[self.length_buckets[0]]
        
        for sample in samples:
            token_count = len(ref_packer.tokenize(sample))
            bucket = self._select_bucket(token_count)
            buckets[bucket].append(sample)
            
        # Pack each bucket
        for bucket_size, bucket_samples in buckets.items():
            if bucket_samples:
                logger.info(f"Packing {len(bucket_samples)} samples into {bucket_size}-token sequences")
                yield from self.packers[bucket_size].pack_samples(bucket_samples)


def demo_packer():
    """Demonstrate sequence packing."""
    print("Sequence Packer Demo")
    print("=" * 50)
    
    # Sample texts of varying lengths
    samples = [
        "def calculate_rsi(prices):\n    return talib.RSI(prices, 14)",
        "import numpy as np\n\nclass Strategy:\n    def __init__(self):\n        self.position = 0\n\n    def calculate_signal(self, prices):\n        returns = np.diff(prices) / prices[:-1]\n        sharpe = returns.mean() / returns.std()\n        return 1 if sharpe > 0.5 else -1",
        "# Quick helper\nx = 1",
    ] * 10  # Repeat for more samples
    
    try:
        packer = SequencePacker()
        
        print(f"\nPacking {len(samples)} samples...")
        
        packed_sequences = list(packer.pack_samples(samples, shuffle=False))
        
        print(f"\nResults:")
        print(f"  Sequences created: {len(packed_sequences)}")
        print(f"  Stats: {packer.stats}")
        
        if packed_sequences:
            seq = packed_sequences[0]
            print(f"\nFirst sequence:")
            print(f"  Samples: {seq.num_samples}")
            print(f"  Tokens: {seq.total_tokens}")
            print(f"  Utilization: {seq.utilization:.1%}")
            
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure transformers is installed: pip install transformers")


if __name__ == "__main__":
    demo_packer()



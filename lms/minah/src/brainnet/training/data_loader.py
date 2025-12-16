"""
Data Loader for Quant Training Datasets.

Loads and streams:
- The Stack v1.2 (bigcode/the-stack) - 3TB Python/TS code
- AlphaFin CoT - Chain-of-thought finance annotations
- Evol-Instruct (SurgeGlobal) - Instruction amplification

HuggingFace references:
- https://huggingface.co/datasets/bigcode/the-stack
- https://huggingface.co/datasets/SurgeGlobal/Evol-Instruct
- https://github.com/AlphaFin-proj/AlphaFin
"""

import logging
from typing import Iterator, Dict, Any, Optional, List, Generator
from dataclasses import dataclass
from itertools import chain, cycle, islice

try:
    from datasets import load_dataset, interleave_datasets, IterableDataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    logging.warning("HuggingFace datasets not installed. Run: pip install datasets")

from .config import DatasetConfig, TrainingConfig

logger = logging.getLogger(__name__)


@dataclass
class DataSample:
    """Unified sample format for all datasets."""
    content: str
    source: str  # 'the_stack', 'alphafin', 'evol_instruct', 'synthetic'
    language: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class QuantDatasetLoader:
    """
    Unified loader for quant training datasets.
    
    Handles streaming from HuggingFace Hub with proper interleaving
    and ratio-based mixing for hybrid learning.
    """
    
    def __init__(self, config: Optional[TrainingConfig] = None):
        if not HF_AVAILABLE:
            raise RuntimeError(
                "HuggingFace datasets required. Install with: pip install datasets"
            )
        self.config = config or TrainingConfig()
        self.dataset_config = self.config.dataset
        self._datasets = {}
        
    def load_the_stack(
        self,
        languages: Optional[List[str]] = None,
        streaming: bool = True
    ) -> IterableDataset:
        """
        Load The Stack v1.2 dataset (bigcode/the-stack).
        
        The Stack contains 6TB of permissively-licensed source code
        covering 358 programming languages. We focus on Python and TS
        for quant code generation.
        
        Args:
            languages: List of languages to load (default: python, typescript)
            streaming: Use streaming mode (essential for 3TB dataset)
            
        Returns:
            IterableDataset for streaming iteration
            
        Example:
            >>> loader = QuantDatasetLoader()
            >>> ds = loader.load_the_stack(languages=["python"])
            >>> for sample in ds.take(10):
            ...     print(sample["content"][:100])
        """
        languages = languages or self.dataset_config.the_stack_languages
        
        logger.info(f"Loading The Stack v1.2 for languages: {languages}")
        
        datasets = []
        for lang in languages:
            try:
                ds = load_dataset(
                    self.dataset_config.the_stack_name,
                    data_dir=f"data/{lang}",
                    split=self.dataset_config.the_stack_split,
                    streaming=streaming,
                    trust_remote_code=True,
                )
                datasets.append(ds)
                logger.info(f"Loaded The Stack - {lang}")
            except Exception as e:
                logger.warning(f"Failed to load The Stack for {lang}: {e}")
                
        if not datasets:
            raise RuntimeError("Failed to load any The Stack languages")
            
        # Interleave multiple languages
        if len(datasets) > 1:
            combined = interleave_datasets(datasets)
        else:
            combined = datasets[0]
            
        self._datasets['the_stack'] = combined
        return combined
    
    def load_alphafin(self, streaming: bool = True) -> IterableDataset:
        """
        Load AlphaFin CoT dataset for finance reasoning.
        
        AlphaFin provides chain-of-thought annotations for:
        - Trading task decomposition
        - Financial reasoning patterns
        - Temporal/sequential decision making
        
        Note: The exact HF path may vary. Check:
        https://github.com/AlphaFin-proj/AlphaFin
        
        Returns:
            IterableDataset with CoT annotations
        """
        logger.info("Loading AlphaFin CoT dataset")
        
        try:
            # Try HuggingFace Hub first
            ds = load_dataset(
                self.dataset_config.alphafin_name,
                split=self.dataset_config.alphafin_split,
                streaming=streaming,
                trust_remote_code=True,
            )
            logger.info("Loaded AlphaFin from HuggingFace Hub")
        except Exception as e:
            logger.warning(f"AlphaFin not found on HF Hub: {e}")
            logger.info("Attempting to load from GitHub...")
            
            # Fallback: Try loading from local or GitHub clone
            try:
                ds = load_dataset(
                    "json",
                    data_files="data/alphafin/*.json",
                    split="train",
                    streaming=streaming,
                )
            except Exception as e2:
                logger.error(f"Failed to load AlphaFin: {e2}")
                logger.info(
                    "Clone AlphaFin repo: git clone https://github.com/AlphaFin-proj/AlphaFin"
                )
                raise
                
        self._datasets['alphafin'] = ds
        return ds
    
    def load_evol_instruct(self, streaming: bool = True) -> IterableDataset:
        """
        Load Evol-Instruct dataset for instruction amplification.
        
        Evol-Instruct generates evolved instruction pairs for:
        - More complex reasoning chains
        - Diverse problem formulations
        - Enhanced instruction-following
        
        Reference: https://huggingface.co/datasets/SurgeGlobal/Evol-Instruct
        
        Returns:
            IterableDataset with instruction pairs
        """
        logger.info("Loading Evol-Instruct dataset")
        
        try:
            ds = load_dataset(
                self.dataset_config.evol_instruct_name,
                split=self.dataset_config.evol_instruct_split,
                streaming=streaming,
                trust_remote_code=True,
            )
            logger.info("Loaded Evol-Instruct successfully")
        except Exception as e:
            logger.error(f"Failed to load Evol-Instruct: {e}")
            raise
            
        self._datasets['evol_instruct'] = ds
        return ds
    
    def create_mixed_dataset(
        self,
        bigcode_ratio: Optional[float] = None,
        alphafin_ratio: Optional[float] = None,
        shuffle: bool = True,
        seed: int = 42,
    ) -> Generator[DataSample, None, None]:
        """
        Create a mixed dataset with configurable ratios.
        
        Default: 50% bigcode (code syntax), 40% AlphaFin (finance reasoning),
        10% reserved for synthetic pairs.
        
        Args:
            bigcode_ratio: Ratio of The Stack samples
            alphafin_ratio: Ratio of AlphaFin samples
            shuffle: Shuffle the interleaved stream
            seed: Random seed for reproducibility
            
        Yields:
            DataSample objects from mixed sources
        """
        bigcode_ratio = bigcode_ratio or self.dataset_config.bigcode_ratio
        alphafin_ratio = alphafin_ratio or self.dataset_config.alphafin_ratio
        
        # Ensure datasets are loaded
        if 'the_stack' not in self._datasets:
            self.load_the_stack()
        if 'alphafin' not in self._datasets:
            try:
                self.load_alphafin()
            except Exception:
                logger.warning("AlphaFin not available, using only The Stack")
                alphafin_ratio = 0
                bigcode_ratio = 1.0
                
        # Calculate sampling probabilities
        total = bigcode_ratio + alphafin_ratio
        p_bigcode = bigcode_ratio / total
        
        import random
        random.seed(seed)
        
        stack_iter = iter(self._datasets['the_stack'])
        alphafin_iter = iter(self._datasets.get('alphafin', []))
        
        while True:
            try:
                if random.random() < p_bigcode:
                    sample = next(stack_iter)
                    yield DataSample(
                        content=sample.get('content', ''),
                        source='the_stack',
                        language=sample.get('lang', 'python'),
                        metadata={
                            'size': sample.get('size'),
                            'ext': sample.get('ext'),
                            'repo': sample.get('max_stars_repo_name'),
                        }
                    )
                else:
                    sample = next(alphafin_iter)
                    yield DataSample(
                        content=self._format_alphafin_sample(sample),
                        source='alphafin',
                        language=None,
                        metadata=sample,
                    )
            except StopIteration:
                break
    
    def _format_alphafin_sample(self, sample: Dict[str, Any]) -> str:
        """Format AlphaFin sample into training format."""
        # Adapt based on actual AlphaFin schema
        if 'instruction' in sample and 'output' in sample:
            return f"### Instruction:\n{sample['instruction']}\n\n### Response:\n{sample['output']}"
        elif 'question' in sample and 'answer' in sample:
            return f"### Question:\n{sample['question']}\n\n### Answer:\n{sample['answer']}"
        elif 'text' in sample:
            return sample['text']
        else:
            return str(sample)
    
    def stream_for_training(
        self,
        max_samples: Optional[int] = None,
        apply_filter: bool = True,
    ) -> Generator[Dict[str, str], None, None]:
        """
        Stream samples in training format.
        
        Args:
            max_samples: Maximum number of samples to yield
            apply_filter: Apply quant keyword filtering
            
        Yields:
            Dict with 'text' key for training
        """
        from .filters import QuantCodeFilter
        
        filter_fn = QuantCodeFilter(self.config) if apply_filter else None
        count = 0
        
        for sample in self.create_mixed_dataset():
            if max_samples and count >= max_samples:
                break
                
            if filter_fn and not filter_fn.should_include(sample.content):
                continue
                
            yield {"text": sample.content}
            count += 1
            
            if count % 10000 == 0:
                logger.info(f"Streamed {count} samples")


def quick_test():
    """Quick test of dataset loading."""
    print("Testing QuantDatasetLoader...")
    
    loader = QuantDatasetLoader()
    
    # Test The Stack loading
    print("\n1. Loading The Stack (Python)...")
    try:
        ds = loader.load_the_stack(languages=["python"])
        sample = next(iter(ds))
        print(f"   ✓ Loaded sample with {len(sample.get('content', ''))} chars")
        print(f"   Keys: {list(sample.keys())}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        
    # Test Evol-Instruct
    print("\n2. Loading Evol-Instruct...")
    try:
        ds = loader.load_evol_instruct()
        sample = next(iter(ds))
        print(f"   ✓ Loaded sample")
        print(f"   Keys: {list(sample.keys())}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        
    print("\nDone!")


if __name__ == "__main__":
    quick_test()



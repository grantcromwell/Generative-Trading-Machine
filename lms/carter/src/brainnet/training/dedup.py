"""
MinHash Deduplication for Training Data.

Implements near-duplicate detection using MinHash LSH:
- Character n-gram based fingerprinting
- Locality-Sensitive Hashing for efficient similarity search
- Configurable Jaccard similarity threshold

Based on techniques from:
- Awesome-Code-LLM deduplication
- The Stack dataset processing pipeline
"""

import hashlib
import logging
from typing import List, Set, Iterator, Tuple, Optional, Generator
from dataclasses import dataclass, field
import struct

from .config import DeduplicationConfig, TrainingConfig

logger = logging.getLogger(__name__)


def _hash_func(seed: int):
    """Create a hash function with given seed."""
    def _hash(data: bytes) -> int:
        return int(hashlib.sha256(data + struct.pack('I', seed)).hexdigest()[:16], 16)
    return _hash


class MinHashSignature:
    """
    MinHash signature for near-duplicate detection.
    
    Uses multiple hash functions to create a signature
    that approximates Jaccard similarity.
    """
    
    def __init__(self, num_perm: int = 128, seed: int = 42):
        """
        Initialize MinHash signature generator.
        
        Args:
            num_perm: Number of hash permutations (more = more accurate)
            seed: Random seed for reproducibility
        """
        self.num_perm = num_perm
        self.seed = seed
        
        # Pre-generate hash functions
        self._hash_funcs = [_hash_func(seed + i) for i in range(num_perm)]
        
        # Initialize signature with max values
        self._signature = [float('inf')] * num_perm
        
    def update(self, token: str) -> None:
        """Update signature with a token."""
        token_bytes = token.encode('utf-8')
        for i, hash_func in enumerate(self._hash_funcs):
            hash_val = hash_func(token_bytes)
            self._signature[i] = min(self._signature[i], hash_val)
            
    def update_batch(self, tokens: List[str]) -> None:
        """Update signature with multiple tokens."""
        for token in tokens:
            self.update(token)
            
    @property
    def signature(self) -> Tuple[int, ...]:
        """Get the signature as a tuple (hashable)."""
        return tuple(int(x) if x != float('inf') else 0 for x in self._signature)
    
    def jaccard_similarity(self, other: 'MinHashSignature') -> float:
        """
        Estimate Jaccard similarity with another signature.
        
        Args:
            other: Another MinHashSignature
            
        Returns:
            Estimated Jaccard similarity (0.0 to 1.0)
        """
        if self.num_perm != other.num_perm:
            raise ValueError("Signatures must have same num_perm")
            
        matches = sum(
            1 for a, b in zip(self._signature, other._signature)
            if a == b
        )
        return matches / self.num_perm


class LSHIndex:
    """
    Locality-Sensitive Hashing index for efficient similarity search.
    
    Partitions signatures into bands for O(1) candidate lookup.
    """
    
    def __init__(self, num_perm: int = 128, threshold: float = 0.7):
        """
        Initialize LSH index.
        
        Args:
            num_perm: Number of MinHash permutations
            threshold: Jaccard similarity threshold for near-duplicates
        """
        self.num_perm = num_perm
        self.threshold = threshold
        
        # Calculate optimal band configuration
        # For threshold t, bands b, rows r: t ≈ (1/b)^(1/r)
        self.num_bands, self.rows_per_band = self._optimal_params(
            num_perm, threshold
        )
        
        # Hash tables for each band
        self._buckets: List[dict] = [{} for _ in range(self.num_bands)]
        
        # Store all signatures
        self._signatures: dict = {}
        
    def _optimal_params(
        self, num_perm: int, threshold: float
    ) -> Tuple[int, int]:
        """Calculate optimal bands and rows for threshold."""
        best_bands = 1
        best_rows = num_perm
        best_error = float('inf')
        
        for bands in range(1, num_perm + 1):
            if num_perm % bands != 0:
                continue
            rows = num_perm // bands
            
            # Probability of being candidate: 1 - (1 - t^r)^b
            prob = 1 - (1 - threshold ** rows) ** bands
            error = abs(prob - 0.5)
            
            if error < best_error:
                best_error = error
                best_bands = bands
                best_rows = rows
                
        logger.debug(f"LSH params: {best_bands} bands, {best_rows} rows")
        return best_bands, best_rows
    
    def add(self, doc_id: str, signature: Tuple[int, ...]) -> None:
        """
        Add a document signature to the index.
        
        Args:
            doc_id: Unique document identifier
            signature: MinHash signature tuple
        """
        self._signatures[doc_id] = signature
        
        # Add to each band's bucket
        for band_idx in range(self.num_bands):
            start = band_idx * self.rows_per_band
            end = start + self.rows_per_band
            band_hash = hash(signature[start:end])
            
            if band_hash not in self._buckets[band_idx]:
                self._buckets[band_idx][band_hash] = set()
            self._buckets[band_idx][band_hash].add(doc_id)
    
    def query(self, signature: Tuple[int, ...]) -> Set[str]:
        """
        Find candidate near-duplicates.
        
        Args:
            signature: MinHash signature to query
            
        Returns:
            Set of candidate document IDs
        """
        candidates = set()
        
        for band_idx in range(self.num_bands):
            start = band_idx * self.rows_per_band
            end = start + self.rows_per_band
            band_hash = hash(signature[start:end])
            
            if band_hash in self._buckets[band_idx]:
                candidates.update(self._buckets[band_idx][band_hash])
                
        return candidates
    
    def is_duplicate(
        self, signature: Tuple[int, ...], verify: bool = True
    ) -> bool:
        """
        Check if signature is a near-duplicate of existing documents.
        
        Args:
            signature: MinHash signature to check
            verify: Verify candidates with actual similarity
            
        Returns:
            True if near-duplicate found
        """
        candidates = self.query(signature)
        
        if not candidates:
            return False
            
        if not verify:
            return True
            
        # Verify with actual similarity
        for doc_id in candidates:
            existing_sig = self._signatures[doc_id]
            similarity = self._signature_similarity(signature, existing_sig)
            if similarity >= self.threshold:
                return True
                
        return False
    
    def _signature_similarity(
        self, sig1: Tuple[int, ...], sig2: Tuple[int, ...]
    ) -> float:
        """Calculate similarity between two signatures."""
        matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
        return matches / len(sig1)


class MinHashDeduplicator:
    """
    Complete deduplication pipeline for training data.
    
    Usage:
        dedup = MinHashDeduplicator()
        
        for doc in documents:
            if not dedup.is_duplicate(doc):
                dedup.add(doc)
                yield doc
    """
    
    def __init__(self, config: Optional[TrainingConfig] = None):
        """
        Initialize deduplicator.
        
        Args:
            config: Training configuration with dedup settings
        """
        self.config = config or TrainingConfig()
        dedup_config = self.config.dedup
        
        self.num_perm = dedup_config.num_perm
        self.threshold = dedup_config.threshold
        self.ngram_size = dedup_config.ngram_size
        self.seed = dedup_config.seed
        
        # Initialize LSH index
        self.lsh = LSHIndex(self.num_perm, self.threshold)
        
        # Stats
        self._total_processed = 0
        self._duplicates_found = 0
        self._doc_counter = 0
        
    def _get_ngrams(self, text: str) -> List[str]:
        """Extract character n-grams from text."""
        # Normalize: lowercase, collapse whitespace
        text = ' '.join(text.lower().split())
        
        if len(text) < self.ngram_size:
            return [text]
            
        return [
            text[i:i + self.ngram_size]
            for i in range(len(text) - self.ngram_size + 1)
        ]
    
    def _compute_signature(self, text: str) -> Tuple[int, ...]:
        """Compute MinHash signature for text."""
        minhash = MinHashSignature(self.num_perm, self.seed)
        ngrams = self._get_ngrams(text)
        minhash.update_batch(ngrams)
        return minhash.signature
    
    def is_duplicate(self, text: str) -> bool:
        """
        Check if text is a near-duplicate.
        
        Args:
            text: Document text to check
            
        Returns:
            True if near-duplicate exists in index
        """
        self._total_processed += 1
        signature = self._compute_signature(text)
        
        if self.lsh.is_duplicate(signature):
            self._duplicates_found += 1
            return True
        return False
    
    def add(self, text: str, doc_id: Optional[str] = None) -> str:
        """
        Add text to the index.
        
        Args:
            text: Document text
            doc_id: Optional document ID (auto-generated if not provided)
            
        Returns:
            Document ID
        """
        if doc_id is None:
            doc_id = f"doc_{self._doc_counter}"
            self._doc_counter += 1
            
        signature = self._compute_signature(text)
        self.lsh.add(doc_id, signature)
        return doc_id
    
    def process(self, text: str, doc_id: Optional[str] = None) -> bool:
        """
        Process text: check and add if not duplicate.
        
        Args:
            text: Document text
            doc_id: Optional document ID
            
        Returns:
            True if text is unique (added), False if duplicate (skipped)
        """
        if self.is_duplicate(text):
            return False
        self.add(text, doc_id)
        return True
    
    def deduplicate_stream(
        self,
        documents: Iterator[str],
        max_docs: Optional[int] = None,
    ) -> Generator[str, None, None]:
        """
        Deduplicate a stream of documents.
        
        Args:
            documents: Iterator of document texts
            max_docs: Optional maximum documents to process
            
        Yields:
            Unique documents
        """
        count = 0
        
        for doc in documents:
            if max_docs and count >= max_docs:
                break
                
            if self.process(doc):
                yield doc
                count += 1
                
            if self._total_processed % 10000 == 0:
                logger.info(
                    f"Processed {self._total_processed}, "
                    f"duplicates: {self._duplicates_found}, "
                    f"unique: {count}"
                )
    
    @property
    def stats(self) -> dict:
        """Get deduplication statistics."""
        return {
            'total_processed': self._total_processed,
            'duplicates_found': self._duplicates_found,
            'unique_count': self._total_processed - self._duplicates_found,
            'duplicate_rate': (
                self._duplicates_found / self._total_processed
                if self._total_processed > 0 else 0
            ),
        }


def demo_dedup():
    """Demonstrate deduplication."""
    
    docs = [
        "def calculate_sharpe_ratio(returns):\n    return np.mean(returns) / np.std(returns)",
        "def calculate_sharpe_ratio(returns):\n    return np.mean(returns) / np.std(returns)",  # Exact dup
        "def compute_sharpe(ret):\n    return np.mean(ret) / np.std(ret)",  # Near dup
        "def calculate_bollinger_bands(prices, window=20):\n    return prices.rolling(window).mean()",
        "This is completely different content about cooking recipes.",
    ]
    
    dedup = MinHashDeduplicator()
    
    print("MinHash Deduplication Demo")
    print("=" * 50)
    
    for i, doc in enumerate(docs):
        is_dup = dedup.is_duplicate(doc)
        if not is_dup:
            dedup.add(doc, f"doc_{i}")
            status = "✓ UNIQUE"
        else:
            status = "✗ DUPLICATE"
        print(f"\nDoc {i}: {status}")
        print(f"  Preview: {doc[:60]}...")
        
    print(f"\nStats: {dedup.stats}")


if __name__ == "__main__":
    demo_dedup()



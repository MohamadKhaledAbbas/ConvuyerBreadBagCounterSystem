"""
Bidirectional batch override smoother.

Ported from v1 BreadBagCounterSystem - this logic is kept as it handles
low confidence classification scenarios by allowing batch-level corrections.

When a batch has low confidence classifications, this smoother can:
1. Look at the overall batch composition
2. Override individual low-confidence predictions
3. Apply majority voting within confidence-weighted windows
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import time

from src.utils.AppLogging import logger


@dataclass
class ClassificationRecord:
    """Record of a single classification event."""
    track_id: int
    class_name: str
    confidence: float
    vote_ratio: float
    timestamp: float
    batch_id: Optional[int] = None
    smoothed: bool = False
    original_class: Optional[str] = None


@dataclass
class BatchState:
    """State of a classification batch."""
    batch_id: int
    records: List[ClassificationRecord] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    finalized: bool = False
    
    @property
    def class_distribution(self) -> Dict[str, int]:
        """Get count of each class in batch."""
        dist = defaultdict(int)
        for rec in self.records:
            dist[rec.class_name] += 1
        return dict(dist)
    
    @property
    def confidence_weighted_distribution(self) -> Dict[str, float]:
        """Get confidence-weighted class distribution."""
        dist = defaultdict(float)
        for rec in self.records:
            dist[rec.class_name] += rec.confidence
        return dict(dist)


class BidirectionalSmoother:
    """
    Batch-level classification smoother.
    
    Key features:
    1. Accumulates classifications into batches
    2. Identifies low-confidence predictions
    3. Applies bidirectional smoothing within batch
    4. Can override individual predictions based on batch context
    
    Smoothing rules:
    - If a class appears only once with low confidence in a batch
      dominated by another class, it may be overridden
    - Confidence threshold determines what counts as "low"
    - Vote ratio threshold determines batch dominance
    """
    
    def __init__(
        self,
        confidence_threshold: float = 0.7,
        vote_ratio_threshold: float = 0.6,
        batch_size: int = 10,
        batch_timeout_seconds: float = 30.0,
        min_batch_dominance: float = 0.7
    ):
        """
        Initialize smoother.
        
        Args:
            confidence_threshold: Below this, classification is considered uncertain
            vote_ratio_threshold: Below this, evidence ratio is considered weak
            batch_size: Target batch size for smoothing
            batch_timeout_seconds: Force finalize batch after this duration
            min_batch_dominance: Min ratio for a class to be considered dominant
        """
        self.confidence_threshold = confidence_threshold
        self.vote_ratio_threshold = vote_ratio_threshold
        self.batch_size = batch_size
        self.batch_timeout_seconds = batch_timeout_seconds
        self.min_batch_dominance = min_batch_dominance
        
        # Current batch
        self.current_batch: Optional[BatchState] = None
        self._next_batch_id = 1
        
        # Completed batches history
        self.completed_batches: List[BatchState] = []
        
        # Smoothing statistics
        self.total_records = 0
        self.smoothed_records = 0
        
        logger.info(
            f"[BidirectionalSmoother] Initialized with confidence_threshold={confidence_threshold}, "
            f"batch_size={batch_size}"
        )
    
    def _create_batch(self) -> BatchState:
        """Create a new batch."""
        batch = BatchState(batch_id=self._next_batch_id)
        self._next_batch_id += 1
        return batch
    
    def _is_low_confidence(self, record: ClassificationRecord) -> bool:
        """Check if a record has low confidence."""
        return (
            record.confidence < self.confidence_threshold or
            record.vote_ratio < self.vote_ratio_threshold
        )
    
    def _apply_batch_smoothing(self, batch: BatchState) -> List[ClassificationRecord]:
        """
        Apply bidirectional smoothing to a batch.
        
        Returns:
            List of potentially modified records
        """
        if len(batch.records) < 2:
            return batch.records
        
        # Find dominant class
        weighted_dist = batch.confidence_weighted_distribution
        total_weight = sum(weighted_dist.values())
        
        if total_weight == 0:
            return batch.records
        
        dominant_class = max(weighted_dist, key=weighted_dist.get)
        dominance_ratio = weighted_dist[dominant_class] / total_weight
        
        # Check if there's a clearly dominant class
        if dominance_ratio < self.min_batch_dominance:
            logger.debug(
                f"[BidirectionalSmoother] Batch {batch.batch_id} has no dominant class "
                f"(best: {dominant_class} at {dominance_ratio:.2f})"
            )
            return batch.records
        
        # Apply smoothing to low-confidence outliers
        smoothed_records = []
        
        for record in batch.records:
            if (
                self._is_low_confidence(record) and
                record.class_name != dominant_class
            ):
                # This is a low-confidence outlier - consider overriding
                # Check if it's truly an outlier (only one of its class)
                class_count = batch.class_distribution.get(record.class_name, 0)
                
                if class_count == 1:
                    # Single occurrence with low confidence - override
                    smoothed = ClassificationRecord(
                        track_id=record.track_id,
                        class_name=dominant_class,
                        confidence=record.confidence,  # Keep original confidence
                        vote_ratio=record.vote_ratio,
                        timestamp=record.timestamp,
                        batch_id=batch.batch_id,
                        smoothed=True,
                        original_class=record.class_name
                    )
                    smoothed_records.append(smoothed)
                    self.smoothed_records += 1
                    
                    logger.info(
                        f"[BidirectionalSmoother] Track {record.track_id}: "
                        f"Smoothed {record.class_name} -> {dominant_class} "
                        f"(conf={record.confidence:.2f}, batch dominance={dominance_ratio:.2f})"
                    )
                else:
                    smoothed_records.append(record)
            else:
                smoothed_records.append(record)
        
        return smoothed_records
    
    def add_classification(
        self,
        track_id: int,
        class_name: str,
        confidence: float,
        vote_ratio: float
    ) -> Optional[List[ClassificationRecord]]:
        """
        Add a classification to the current batch.
        
        Args:
            track_id: Track identifier
            class_name: Predicted class
            confidence: Classification confidence
            vote_ratio: Vote ratio from evidence accumulation
            
        Returns:
            List of smoothed records if batch was finalized, None otherwise
        """
        self.total_records += 1
        
        # Create batch if needed
        if self.current_batch is None:
            self.current_batch = self._create_batch()
        
        # Create record
        record = ClassificationRecord(
            track_id=track_id,
            class_name=class_name,
            confidence=confidence,
            vote_ratio=vote_ratio,
            timestamp=time.time(),
            batch_id=self.current_batch.batch_id
        )
        
        self.current_batch.records.append(record)
        
        # Check if batch should be finalized
        should_finalize = (
            len(self.current_batch.records) >= self.batch_size or
            (time.time() - self.current_batch.created_at) > self.batch_timeout_seconds
        )
        
        if should_finalize:
            return self.finalize_batch()
        
        return None
    
    def finalize_batch(self) -> List[ClassificationRecord]:
        """
        Finalize current batch and apply smoothing.
        
        Returns:
            List of smoothed classification records
        """
        if self.current_batch is None or not self.current_batch.records:
            return []
        
        # Apply smoothing
        smoothed = self._apply_batch_smoothing(self.current_batch)
        
        # Mark as finalized
        self.current_batch.finalized = True
        self.completed_batches.append(self.current_batch)
        
        # Limit history
        if len(self.completed_batches) > 100:
            self.completed_batches = self.completed_batches[-100:]
        
        logger.info(
            f"[BidirectionalSmoother] Batch {self.current_batch.batch_id} finalized: "
            f"{len(smoothed)} records, distribution: {self.current_batch.class_distribution}"
        )
        
        # Reset for next batch
        self.current_batch = None
        
        return smoothed
    
    def get_pending_records(self) -> List[ClassificationRecord]:
        """Get records in current unfinalied batch."""
        if self.current_batch is None:
            return []
        return self.current_batch.records.copy()
    
    def get_statistics(self) -> Dict[str, any]:
        """Get smoothing statistics."""
        return {
            'total_records': self.total_records,
            'smoothed_records': self.smoothed_records,
            'smoothing_rate': (
                self.smoothed_records / self.total_records
                if self.total_records > 0 else 0.0
            ),
            'completed_batches': len(self.completed_batches),
            'pending_records': len(self.get_pending_records())
        }
    
    def cleanup(self):
        """Clean up resources."""
        if self.current_batch and self.current_batch.records:
            # Force finalize remaining batch
            self.finalize_batch()
        
        self.completed_batches.clear()
        logger.info(
            f"[BidirectionalSmoother] Cleanup complete, "
            f"smoothed {self.smoothed_records}/{self.total_records} records"
        )

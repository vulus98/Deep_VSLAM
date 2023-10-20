from abc import ABC, abstractmethod
from typing import List
import torch
from torch.utils.data.dataset import IterableDataset

"""
IterableDataGenerator is an IterableDataset subclass that gives synthetic data.

abstract methods (all used in __next__):
_next_correspondences: creates next item for iteration
_fixed_size: makes sure there are exactly _n_correspondences in the item
_normalize_correspondences: normalizes correspondences if self._normalize
_visualize_correspondences: visualizes correspondences if self._visualize

"""

class IterableDataGenerator(IterableDataset, ABC):
    def __init__(self, n_correspondences: int, image_dim: List[int], intrinsic_matrix: torch.Tensor, normalize: bool = False, visualize: bool = False):
        self._n_correspondences = n_correspondences
        self._image_dim = image_dim
        self._k_matrix = intrinsic_matrix
        self._normalize = normalize
        self._visualize = visualize

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        correspondences = self._next_correspondences()
        correspondences = self._fixed_size(correspondences, self._n_correspondences)
        if self._visualize:
            self._visualize_correspondences(correspondences)
        if self._normalize:
            correspondences = self._normalize_correspondences(correspondences)
        return correspondences

    @abstractmethod
    def _next_correspondences(self):
        """
        Method should return dictionary of correspondances' data. 
        """
        pass

    @abstractmethod
    def _fixed_size(self,):
        """
        Method should bring number of correspondances to a fixed size. (either by padding or cutting)
        """
        pass

    @abstractmethod
    def _normalize_correspondences(self,):
        """
        Method should normalize correspondences according to data type.
        """
        pass

    @abstractmethod
    def _visualize_correspondences(self,):
        """
        Method should visualize the item given by __next__(self)
        """
        pass
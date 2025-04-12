import random
from typing import List, Optional
from torch.utils.data.sampler import Sampler
from torch.utils.data import Dataset
from dataclasses import dataclass


class RandomDomainSampler(Sampler):
    def __init__(
        self,
        dataset: Dataset,
        domain_labels: List[int],
        batch_size: int = 1,
        n_domains_per_sample: int = 2,
        shuffle: bool = True,
        consistent_iterating: bool = False,
    ):
        """Sample data from multiple domains in a balanced way. If domains have
        different number of samples, the number of samples will be the minimum
        number of samples for each domain.

        Parameters
        ----------
        dataset : Dataset
            The dataset to sample from.
        domain_labels : List[int]
            The domain labels for each sample in the dataset.
        batch_size : int, optional
            The number of samples for each domain in a batch, by default 1.
            The effective batch size will be batch_size * n_domains_per_sample.
        n_domains_per_sample : int, optional
            The number of domains to sample from in each batch, by default 2.
            Note that, the domain labels must have at least n_domains_per_sample
            distinct domains.
        shuffle : bool, optional
            Shuffle the samples in each domain before sampling, by default True
        consistent_iterating : bool, optional
            As each domain may have different number of samples, in different
            iterations, the same samples may not be used. If True, the same
            samples will be used in every iteration, by default False.
        """
        self.dataset = dataset
        self.domain_labels = domain_labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.consistent_iterating = consistent_iterating
        self.domains = set(domain_labels)
        self.min_batches = min(
            len([l for l in domain_labels if l == d]) // batch_size
            for d in self.domains
        )
        self.n_domains_per_sample = n_domains_per_sample
        assert self.min_batches > 0, "Not enough samples for a batch"

        self.cached = None
        self.seed = random.random()
        self.rng = random.Random(self.seed)

    def __len__(self):
        return (self.min_batches * len(self.domains)) // self.n_domains_per_sample

    def _select_samples(self):
        indices = {}
        for d in self.domains:
            idxs = [i for i, l in enumerate(self.domain_labels) if l == d]
            if self.shuffle:
                random.shuffle(idxs)
            idxs = idxs[: self.min_batches * self.batch_size]
            indices[d] = idxs
        return indices

    def __iter__(self):
        if self.consistent_iterating:
            if self.cached is None:

                self.cached = self._select_samples()
            indices = self.cached.copy()
        else:
            indices = self._select_samples()

        batches = []
        if self.consistent_iterating:
            rng = random.Random(self.seed)
        else:
            rng = self.rng

        while True:
            batch = []
            for i in range(self.n_domains_per_sample):
                if len(indices) == 0:
                    break

                selected_domain = rng.choice(list(indices.keys()))
                idxs = indices[selected_domain]

                selected_indices = idxs[: self.batch_size]
                batch += selected_indices

                idxs = idxs[self.batch_size :]
                if len(idxs) < self.batch_size:
                    del indices[selected_domain]
                else:
                    indices[selected_domain] = idxs

            if len(batch) != self.batch_size * self.n_domains_per_sample:
                break

            batches.append(batch)
        yield from batches

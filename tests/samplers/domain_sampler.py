import pytest
from torch.utils.data import Dataset

from minerva.samplers.domain_sampler import RandomDomainSampler


class DummyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


@pytest.fixture
def dummy_dataset():
    data = list(range(100))
    return DummyDataset(data)


@pytest.fixture
def domain_labels():
    # Cycle through 5 domain labels
    return [i // 20 for i in range(100)]


def test_random_domain_sampler_length(dummy_dataset, domain_labels):
    sampler = RandomDomainSampler(
        dummy_dataset, domain_labels, batch_size=2, n_domains_per_sample=2
    )
    assert len(sampler) == 25
    
    
def test_random_domain_sampler_length_2(dummy_dataset, domain_labels):
    sampler = RandomDomainSampler(
        dummy_dataset, domain_labels, batch_size=2, n_domains_per_sample=4
    )
    assert len(sampler) == 12


def test_random_domain_sampler_iteration(dummy_dataset, domain_labels):
    sampler = RandomDomainSampler(
        dummy_dataset, domain_labels, batch_size=2, n_domains_per_sample=2
    )
    batches = list(iter(sampler))
    assert len(batches) == 25
    for batch in batches:
        assert len(batch) == 4


def test_random_domain_sampler_consistent_iterating(
    dummy_dataset, domain_labels
):
    sampler = RandomDomainSampler(
        dummy_dataset,
        domain_labels,
        batch_size=2,
        n_domains_per_sample=2,
        consistent_iterating=True,
    )
    batches1 = list(iter(sampler))
    batches2 = list(iter(sampler))
    assert batches1 == batches2


def test_random_domain_sampler_insufficient_samples(dummy_dataset, domain_labels):
    with pytest.raises(AssertionError, match="Not enough samples for a batch"):
        RandomDomainSampler(
            dummy_dataset, domain_labels, batch_size=100, n_domains_per_sample=2
        )

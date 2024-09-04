import torch
import lightning as L
from minerva.engines.patch_inferencer_engine import WeightedAvgPatchInferencer, VotingPatchInferencer

pyramid = [
    [
        [1, 1, 1, 1, 1],
        [1, 2, 2, 2, 1],
        [1, 2, 3, 2, 1],
        [1, 2, 2, 2, 1],
        [1, 1, 1, 1, 1],
    ]
]

weights = [
    [
        [0.25, 0.25, 0.25, 0.25, 0.25],
        [0.25, 0.50, 0.50, 0.50, 0.25],
        [0.25, 0.50, 1.00, 0.50, 0.25],
        [0.25, 0.50, 0.50, 0.50, 0.25],
        [0.25, 0.25, 0.25, 0.25, 0.25],
    ]
]

classes = [
    [
        [
            [0.50, 0.25, 0.25],
            [0.50, 0.25, 0.25],
            [0.25, 0.50, 0.25],
            [0.25, 0.25, 0.50],
        ],
        [
            [0.50, 0.25, 0.25],
            [0.50, 0.25, 0.25],
            [0.25, 0.50, 0.25],
            [0.25, 0.25, 0.50],
        ],
        [
            [0.50, 0.25, 0.25],
            [0.50, 0.25, 0.25],
            [0.25, 0.50, 0.25],
            [0.25, 0.25, 0.50],
        ],
        [
            [0.50, 0.25, 0.25],
            [0.50, 0.25, 0.25],
            [0.25, 0.50, 0.25],
            [0.25, 0.25, 0.50],
        ],
    ]
]


def weight_function(shape: tuple) -> torch.Tensor:
    assert shape == (1, 5, 5), "Reference shape must be (1, 5, 5)"
    return torch.Tensor(weights)


class Pyramid5(L.LightningModule):
    # Pyramid model, returns a pyramid independent of input
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1:] == (
            1,
            5,
            5,
        ), "Model must receive array of tensors of shape (1, 5, 5)"
        return torch.Tensor(pyramid).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)


class Classifier(L.LightningModule):
    # Classifier model, returns same classification result for 4x4 windows with 3 classes
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1:] == (
            1,
            4,
            4,
        ), "Model must receive array of tensors of shape (1, 4, 4)"
        return torch.Tensor(classes).unsqueeze(0).repeat(x.shape[0], 1, 1, 1, 1)


def test_weighted_avg_patch_inferencer_basic():
    # Test WeightedAvgPatchInferencer basic usage
    model = Pyramid5()
    inferencer_no_pad = WeightedAvgPatchInferencer(model=model, input_shape=(1, 5, 5))

    model_input = torch.zeros((2, 20, 20))

    output = inferencer_no_pad(model_input)
    assert model_input.shape == output.shape, "Input and Output don't have same shape"
    expected_output = torch.Tensor(pyramid).repeat(2, 4, 4)

    assert torch.equal(
        output, expected_output
    ), "Output does not match with expected output"


def test_weighted_avg_patch_inferencer():
    # Test WeightedAvgPatchInferencer with offsets, padding and custom weight function
    model = Pyramid5()
    inferencer = WeightedAvgPatchInferencer(
        model=model,
        weight_function=weight_function,
        input_shape=(1, 5, 5),
        offsets=[
            (0, -1, 0),
            (0, -2, 0),
            (0, -3, 0),
            (0, -4, 0),
            (0, 0, -1),
            (0, -1, -1),
            (0, -2, -1),
            (0, -3, -1),
            (0, -4, -1),
            (0, 0, -2),
            (0, -1, -2),
            (0, -2, -2),
            (0, -3, -2),
            (0, -4, -2),
            (0, 0, -3),
            (0, -1, -3),
            (0, -2, -3),
            (0, -3, -3),
            (0, -4, -3),
            (0, 0, -4),
            (0, -1, -4),
            (0, -2, -4),
            (0, -3, -4),
            (0, -4, -4),
        ],
        padding={"pad": (0, 4, 4)},
    )

    model_input = torch.zeros((2, 20, 20))

    output = inferencer(model_input)
    assert model_input.shape == output.shape, "Input and Output don't have same shape"
    # Offsets used in this test result in each point of the output to be the combination of all the 25 points 
    # that make up the 5x5 pyramid, using the weights to combine them in a weighted average 
    expected_value = torch.sum(
        torch.Tensor(pyramid) * torch.Tensor(weights)
    ) / torch.sum(torch.Tensor(weights))
    expected_output = torch.full((2, 20, 20), fill_value=expected_value)

    assert torch.equal(
        output, expected_output
    ), "Output does not match with expected output"


def test_voting_patch_inferencer_basic():
    # Test VotingPatchInferencer basic usage
    model = Classifier()
    inferencer = VotingPatchInferencer(
        model=model, input_shape=(1, 4, 4), num_classes=3, voting="soft"
    )

    model_input = torch.zeros((2, 20, 20))

    output = inferencer(model_input)
    assert model_input.shape == output.shape, "Input and Output don't have same shape"
    expected_output = torch.Tensor(
        [
            [0, 0, 1, 2],
            [0, 0, 1, 2],
            [0, 0, 1, 2],
            [0, 0, 1, 2],
        ]
    ).repeat(2, 5, 5)

    assert torch.equal(
        output, expected_output
    ), "Output does not match with expected output"

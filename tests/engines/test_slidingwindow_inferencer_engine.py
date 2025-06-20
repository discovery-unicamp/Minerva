import torch
from minerva.engines.slidingwindow_inferencer_engine import SlidingWindowInferencer
from minerva.models.nets.base import SimpleSupervisedModel


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
            [0.5, 0.5, 0.25, 0.25],
            [0.5, 0.5, 0.25, 0.25],
            [0.5, 0.5, 0.25, 0.25],
            [0.5, 0.5, 0.25, 0.25],
        ]
    ],
    [
        [
            [0.25, 0.25, 0.5, 0.25],
            [0.25, 0.25, 0.5, 0.25],
            [0.25, 0.25, 0.5, 0.25],
            [0.25, 0.25, 0.5, 0.25],
        ]
    ],
    [
        [
            [0.25, 0.25, 0.25, 0.5],
            [0.25, 0.25, 0.25, 0.5],
            [0.25, 0.25, 0.25, 0.5],
            [0.25, 0.25, 0.25, 0.5],
        ]
    ],
]


def weight_function(shape: tuple) -> torch.Tensor:
    assert shape == (1, 5, 5), "Reference shape must be (1, 5, 5)"
    return torch.Tensor(weights)


class Pyramid5(SimpleSupervisedModel):
    # Pyramid model, returns a pyramid independent of input
    def __init__(self):
        super().__init__(None, None, None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1:] == (
            1,
            5,
            5,
        ), "Model must receive array of tensors of shape (1, 5, 5)"
        return torch.Tensor(pyramid).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)


class Classifier(SimpleSupervisedModel):
    # Classifier model, returns same classification result for 4x4 windows with 3 classes

    def __init__(self):
        super().__init__(None, None, None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1:] == (
            1,
            4,
            4,
        ), "Model must receive array of tensors of shape (1, 4, 4)"
        return torch.Tensor(classes).unsqueeze(0).repeat(x.shape[0], 1, 1, 1, 1)


def test_slidingwindow_inferencer_regression_basic():
    # Test SlidingWindowInferencer basic usage in regression task (same behaviour as PatchInferencer test)
    model = Pyramid5()
    inferencer_no_pad = SlidingWindowInferencer(
        model=model, strides=(1, 5, 5), input_shape=(1, 5, 5)
    )

    model_input_batch = torch.zeros((1, 2, 20, 20))

    output = inferencer_no_pad(model_input_batch)
    assert (
        model_input_batch.shape == output.shape
    ), "Input and Output don't have same shape"
    expected_output = torch.Tensor(pyramid).repeat(2, 4, 4).unsqueeze(0)

    assert torch.equal(
        output, expected_output
    ), "Output does not match with expected output"


def test_slidingwindow_inferencer_regression():
    # Test SlidingWindowInferencer with padding and custom weight function in regression task (same behaviour as PatchInferencer test)
    model = Pyramid5()
    inferencer = SlidingWindowInferencer(
        model=model,
        strides=(1, 1, 1),
        weight_function=weight_function,
        input_shape=(1, 5, 5),
        padding={"pad": (0, 24, 24)},
    )

    model_input_batch = torch.zeros((1, 2, 20, 20))

    output = inferencer(model_input_batch)
    assert (
        model_input_batch.shape == output.shape
    ), "Input and Output don't have same shape"
    # Offsets used in this test result in each point of the output to be the combination of all the 25 points
    # that make up the 5x5 pyramid, using the weights to combine them in a weighted average
    expected_value = torch.sum(
        torch.Tensor(pyramid) * torch.Tensor(weights)
    ) / torch.sum(torch.Tensor(weights))
    expected_output_middle = torch.full((1, 2, 16, 16), fill_value=expected_value)

    assert torch.equal(
        output[:, :, 0, :], torch.full((1, 2, 20), fill_value=1)
    ), "Output upper border region does not match with expected values"

    assert torch.equal(
        output[:, :, :, 0], torch.full((1, 2, 20), fill_value=1)
    ), "Output left border region does not match with expected values"
    assert torch.equal(
        output[:, :, 4:, 4:], expected_output_middle
    ), "Output middle region does not match with expected values"


def test_slidingwindow_inferencer_classification_basic():
    # Test SlidingWindowInferencer basic usage in classification task (same behaviour as PatchInferencer test)
    model = Classifier()
    inferencer = SlidingWindowInferencer(
        model=model, strides=(1, 4, 4), input_shape=(1, 4, 4), output_shape=(3, 1, 4, 4)
    )

    model_input_batch = torch.zeros((1, 2, 20, 20))

    output = inferencer(model_input_batch)

    assert (1, 3, 2, 20, 20) == output.shape, "Output doen't have expected shape"
    expected_classification = (
        torch.Tensor(
            [
                [0, 0, 1, 2],
                [0, 0, 1, 2],
                [0, 0, 1, 2],
                [0, 0, 1, 2],
            ]
        )
        .repeat(2, 5, 5)
        .unsqueeze(0)
    )

    predicted_classes = torch.argmax(output, dim=1, keepdim=False)

    assert torch.equal(
        predicted_classes, expected_classification
    ), "Predicted classes don't match with expected classification"


def test_slidingwindow_inferencer_classification():
    # Test SlidingWindowInferencer with offset in classification task (same behaviour as PatchInferencer test)
    model = Classifier()
    inferencer = SlidingWindowInferencer(
        model=model,
        strides=(1, 2, 2),
        input_shape=(1, 4, 4),
        output_shape=(3, 1, 4, 4),
    )

    model_input_batch = torch.zeros((1, 2, 8, 8))

    output = inferencer(model_input_batch)

    assert (1, 3, 2, 8, 8) == output.shape, "Output doen't have expected shape"
    expected_classification = (
        torch.Tensor(
            [
                [0, 0, 0, 0, 0, 0, 1, 2],
                [0, 0, 0, 0, 0, 0, 1, 2],
                [0, 0, 0, 0, 0, 0, 1, 2],
                [0, 0, 0, 0, 0, 0, 1, 2],
            ]
        )
        .repeat(2, 2, 1)
        .unsqueeze(0)
    )

    predicted_classes = torch.argmax(output, dim=1, keepdim=False)

    assert torch.equal(
        predicted_classes, expected_classification
    ), "Predicted classes don't match with expected classification"

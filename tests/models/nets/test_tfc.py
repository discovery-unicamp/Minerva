from minerva.models.nets.tfc import TFC_Backbone, TFC_PredicionHead, IgnoreWhenBatch1
import torch
from minerva.transforms.tfc import TFC_Transforms


def test_tfc_backbone_forward_default():
    input_shape = (9, 128)
    single_encoding_size = 128
    batch_size = 42

    model = TFC_Backbone(
        input_channels=input_shape[0],
        TS_length=input_shape[1],
        single_encoding_size=single_encoding_size,
    )
    assert model is not None
    x = torch.rand(batch_size, *input_shape)
    model(x)
    y = model.get_representations()
    assert len(y) == 4
    h_time, z_time, h_freq, z_freq = y
    assert len(h_time) == len(z_time) == len(h_freq) == len(z_freq) == batch_size
    assert z_time.shape[-1] + z_freq.shape[-1] == single_encoding_size * 2


def test_tfc_backbone_forward_arbitrary():
    input_shape = (7, 227)
    single_encoding_size = 128
    batch_size = 37

    model = TFC_Backbone(
        input_channels=input_shape[0],
        TS_length=input_shape[1],
        single_encoding_size=single_encoding_size,
    )
    assert model is not None
    x = torch.rand(batch_size, *input_shape)
    model(x)
    y = model.get_representations()
    assert len(y) == 4
    h_time, z_time, h_freq, z_freq = y
    assert len(h_time) == len(z_time) == len(h_freq) == len(z_freq) == batch_size
    assert z_time.shape[-1] + z_freq.shape[-1] == single_encoding_size * 2


def test_tfc_prediction_head():
    single_encoding_size = 4
    batch_size = 5
    num_classes = 10

    model = TFC_PredicionHead(
        num_classes=num_classes, single_encoding_size=single_encoding_size
    )
    assert model is not None
    x = torch.rand(batch_size, single_encoding_size * 2)
    y = model(x)
    assert y.shape == (batch_size, num_classes)


def test_transforms_param():
    transforms = TFC_Transforms()
    backbone = TFC_Backbone(
        input_channels=3, TS_length=128, single_encoding_size=128, transform=transforms
    )
    assert backbone is not None


def test_ignore_batch():
    x = torch.tensor([[[3, 1]]])

    class verify_batch_size(torch.nn.Module):
        def forward(self, x):
            if x.shape[0] == 1:
                return 0

    model = IgnoreWhenBatch1(verify_batch_size())

    assert model(x) == 0

    model2 = IgnoreWhenBatch1(verify_batch_size(), active=True)
    assert model2(x) is x


def test_default_jitter_operations():
    transforms = TFC_Transforms()
    assert (
        transforms.jitter_operation == transforms.proportional_jitter
    ), "Default jitter operation should be proportional_jitter, it is {}".format(
        transforms.jitter_operation
    )


def test_custom_jitter_operations():
    def custom_jitter(x, ratio):
        return x + ratio

    transforms = TFC_Transforms(jitter_function=custom_jitter)
    assert (
        transforms.jitter_operation == custom_jitter
    ), "Custom jitter operation should be set correctly"


def test_default_jitter_ratio():
    transforms = TFC_Transforms()
    assert transforms.jitter_ration == 0.08, "Default jitter ratio should be 0.08"


def test_default_params_by_bacbone():
    transforms = TFC_Transforms()
    backbone = TFC_Backbone(
        input_channels=3, TS_length=128, single_encoding_size=128, transform=transforms
    )
    assert backbone is not None
    assert (
        backbone.transform.jitter_ration == 0.08
    ), "Default jitter ratio should be 0.08"
    assert (
        backbone.transform.jitter_operation == transforms.proportional_jitter
    ), "Default jitter operation should be proportional_jitter, it is {}".format(
        backbone.transform.jitter_operation
    )


def test_custom_params_by_backbone():
    def custom_jitter(x, ratio):
        return x + ratio

    transforms = TFC_Transforms(jitter_ratio=0.1, jitter_function=custom_jitter)
    backbone = TFC_Backbone(
        input_channels=3, TS_length=128, single_encoding_size=128, transform=transforms
    )
    assert backbone is not None
    assert backbone.transform.jitter_ration == 0.1, "Custom jitter ratio should be 0.1"
    assert (
        backbone.transform.jitter_operation == custom_jitter
    ), "Custom jitter operation should be set correctly"

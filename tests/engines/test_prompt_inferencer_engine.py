import pytest
import torch
import numpy as np
import lightning as L
from unittest.mock import MagicMock
from typing import List, Dict, Any, Union
from minerva.engines.prompt_inferencer_engine import (
    PromptInferencer,
    PromptInferencerEngine,
)


# Mock model with expected behavior
class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cpu")
        self.best_idx = 0

    def forward(self, batch_dicts: List[Dict[str, Any]], multimask_output=True):
        return [
            {
                "iou_predictions": torch.tensor([[0.8]]),
                "low_res_logits": torch.randn(1, 1, 64, 64),
                "masks_logits": torch.randn(1, 1, 64, 64),
            }
        ]

    def _loss(self, iou, pred, label):
        return torch.tensor(0.5)


# my custom __call__()
def custom_call_for_dinamic_threshold(self, model: Union[L.LightningModule, torch.nn.Module], batched_input: List[Dict[str, Any]]):  # type: ignore
    """
    Performs inference using facie-centered points and multiple binary thresholds. This method is designed to implement a specific experiment condition in the prompts, which is to always choose the best threshold for each new point.
    For each inference point, test all possible thresholds in `thresholds`, and choose the resulting mask with the highest IoU.

    Parameters
    ----------
    model : Union[pl.LightningModule, torch.nn.Module]
        Trained model that accepts inputs with inference points and returns masks and IoUs.
    batched_input : List[Dict[str, Any]]
        List of samples with image, label and original size. Each item must contain the following keys:
            - 'image': input image
            - 'label': ground truth mask
            - 'original_size': original image size

    Returns
    -------
    List[Dict[float, Dict[int, Dict[str, List]]]]
        List containing one dictionary per image in the batch.
        Each dictionary maps thresholds (float) to another dictionary per facie (int), containing metrics and accumulated information:
            - 'ious': List[float]
            - 'intersections': List[int]
            - 'unions': List[int]
            - 'points': List[Tuple[ndarray, ndarray]]
            - 'mask_logits': List[ndarray]
            - 'losses': List[float]
            - 'best_thresholds': List[float]
    """
    thresholds = [
        round(t, 2) for t in np.arange(0.0, 0.81, 0.1)
    ]  # list of possible thresholds
    all_outputs = []

    for item in batched_input:
        image = item["image"]
        label = item["label"]
        original_size = item["original_size"]
        num_facies = torch.unique(label)
        facie_outputs_per_threshold = {}

        threshold = "best"  # we use 'best' as key in output dictionary
        facie_outputs = {}

        for facie in num_facies:
            region = (label == facie).to(torch.int32).to(model.device)
            if torch.sum(region).item() <= 1:
                continue

            real_label = region
            self.set_points()
            point_type = "positive"

            facie_outputs[int(facie)] = {
                "ious": [],
                "intersections": [],
                "unions": [],
                "points": [],
                "mask_logits": [],
                "label": [],
                "losses": [],
                "best_thresholds": [],  # store the threshold used for each point
            }

            # prev_low_res_logits = None

            for _ in range(self.num_points):
                point_coords, point_labels = self.calculate_center_region(
                    region.cpu().numpy(), point_type
                )
                point_coords_tensor = (
                    torch.tensor(point_coords, dtype=torch.long)
                    .unsqueeze(0)
                    .to(model.device)
                )
                point_labels_tensor = (
                    torch.tensor(point_labels, dtype=torch.long)
                    .unsqueeze(0)
                    .to(model.device)
                )

                batch_dict = {
                    "image": image,
                    "label": real_label,
                    "original_size": original_size,
                    "point_coords": point_coords_tensor,
                    "point_labels": point_labels_tensor,
                }

                # if prev_low_res_logits is not None:
                #     batch_dict['mask_inputs'] = prev_low_res_logits

                outputs = model([batch_dict], multimask_output=self.multimask_output)
                iou_predictions = torch.stack(
                    [output["iou_predictions"].squeeze(0) for output in outputs]
                )
                low_res_logits = torch.stack(
                    [output["low_res_logits"].squeeze(0) for output in outputs]
                )
                masks_logits = torch.stack(
                    [output["masks_logits"].squeeze(0) for output in outputs]
                )
                labels = (real_label).unsqueeze(0)

                # prev_low_res_logits = low_res_logits

                loss_pred = model._loss(iou_predictions, masks_logits, labels)

                sigmoid_masks = torch.sigmoid(masks_logits)

                best_iou = 0.0
                best_mask = None
                best_thresh = None
                best_inter = 0
                best_union = 0

                for t in thresholds:
                    masks = sigmoid_masks > t
                    pred_mask = masks.squeeze(1).bool()
                    gt_mask = labels.bool()
                    inter = int((pred_mask & gt_mask).sum().item())
                    uni = int((pred_mask | gt_mask).sum().item())
                    iou = float(inter / uni) if uni != 0 else 0.0

                    if iou > best_iou:
                        best_iou = iou
                        best_mask = masks.clone()
                        best_thresh = t
                        best_inter = inter
                        best_union = uni

                # updates segmentation based on best threshold found
                diff, new_point_type = self.calculate_diff_label_pred(
                    label=(real_label.cpu().numpy()),
                    pred=best_mask.squeeze().cpu().numpy(),
                )
                region = torch.tensor(diff, dtype=torch.int32).to(model.device)
                point_type = new_point_type

                facie_outputs[int(facie)]["ious"].append(best_iou)
                facie_outputs[int(facie)]["intersections"].append(best_inter)
                facie_outputs[int(facie)]["unions"].append(best_union)
                facie_outputs[int(facie)]["points"].append(
                    (
                        point_coords_tensor.cpu().numpy(),
                        point_labels_tensor.cpu().numpy(),
                    )
                )
                facie_outputs[int(facie)]["mask_logits"].append(
                    masks_logits.cpu().numpy()
                )
                facie_outputs[int(facie)]["label"].append(labels.cpu().numpy())
                facie_outputs[int(facie)]["losses"].append(loss_pred.item())
                facie_outputs[int(facie)]["best_thresholds"].append(best_thresh)

        facie_outputs_per_threshold[threshold] = facie_outputs
        all_outputs.append(facie_outputs_per_threshold)

    return all_outputs


def test_prompt_inferencer_instantiation():
    model = DummyModel()
    wrapper = PromptInferencer(model=model)
    assert isinstance(wrapper.prompt_inferencer, PromptInferencerEngine)
    assert wrapper.num_points == 10


def test_forward_calls_engine(monkeypatch):
    model = DummyModel()
    wrapper = PromptInferencer(model=model)

    dummy_input = [
        {
            "image": torch.randn(3, 64, 64),
            "label": torch.zeros(64, 64),
            "original_size": (64, 64),
        }
    ]

    mock_engine = MagicMock()
    mock_engine.return_value = [{"output": "mocked"}]
    wrapper.prompt_inferencer = mock_engine  # substitui o engine inteiro por um mock

    output = wrapper(dummy_input)

    mock_engine.assert_called_once_with(model, batched_input=dummy_input)
    assert output == [{"output": "mocked"}]


def test_single_step_constructs_batch():
    model = DummyModel()
    wrapper = PromptInferencer(model=model)

    dummy_image = torch.randn(2, 3, 64, 64)
    dummy_label = torch.randint(0, 2, (2, 64, 64))
    batch = (dummy_image, dummy_label)

    mock_engine = MagicMock()
    mock_engine.return_value = [{"fake": "output"}, {"fake": "output"}]  # lista válida

    wrapper.prompt_inferencer = mock_engine

    output = wrapper._single_step(batch, batch_idx=0, step_name="predict")
    assert isinstance(output, list)
    assert len(output) == 2


def test_custom_call_is_used():
    model = DummyModel()

    def custom_method(self, model, batched_input):
        return [{"custom": True}]

    engine = PromptInferencerEngine(custom_call=custom_method)
    result = engine(
        model,
        [
            {
                "image": torch.zeros(3, 64, 64),
                "label": torch.zeros(64, 64),
                "original_size": (64, 64),
            }
        ],
    )
    assert result[0]["custom"] is True


def test_calculate_center_region_returns_point():
    engine = PromptInferencerEngine()
    region = np.zeros((100, 100), dtype=np.uint8)
    region[40:60, 45:55] = 1

    coords, labels = engine.calculate_center_region(region, "positive")
    assert coords.shape == (1, 2)
    assert labels[0] == 1


def test_calculate_diff_label_pred_outputs_correct_type():
    engine = PromptInferencerEngine()

    label = np.zeros((64, 64), dtype=np.uint8)
    pred = np.zeros((64, 64), dtype=np.uint8)
    label[10:30, 10:30] = 1
    pred[15:25, 15:25] = 1

    diff, point_type = engine.calculate_diff_label_pred(label, pred)
    assert diff.shape == (64, 64)
    assert point_type in ["positive", "negative"]


def test_custom_call_with_threshold_selection():
    model = DummyModel()

    # Wrapper that enables custom_call
    engine = PromptInferencerEngine(custom_call=custom_call_for_dinamic_threshold)

    # Synthetic data: 1 3x64x64 image with a circular mask
    image = torch.randn(3, 64, 64)
    label = torch.zeros(64, 64, dtype=torch.int32)
    label[20:45, 20:45] = 1  # create a "face" in the center

    batched_input = [{"image": image, "label": label, "original_size": (64, 64)}]

    # execute inference
    result = engine(model, batched_input)

    # checks
    assert isinstance(result, list)
    assert "best" in result[0]  # key used for identify thresholds
    facie_outputs = result[0]["best"]

    assert len(facie_outputs) > 0  # There must be at least 1 facie
    for facie_data in facie_outputs.values():
        assert "best_thresholds" in facie_data
        assert len(facie_data["best_thresholds"]) > 0
        assert all(isinstance(t, float) for t in facie_data["best_thresholds"])

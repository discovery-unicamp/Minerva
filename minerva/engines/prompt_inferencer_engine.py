import cv2
import torch
import numpy as np
import lightning as L
from minerva.engines.engine import _Engine
from typing import Callable, Optional, List, Any, Dict, Union


class PromptInferencer(L.LightningModule):
    def __init__(
        self,
        model: L.LightningModule,
        num_facies: int = 6,
        multimask_output: bool = False,
        num_points: int = 10,
    ):
        """
        This class acts as a normal `L.LightningModule` that wraps a `Sam` model allowing it to perform inference with points.
        This will be useful when the model execute inference using points that oriented the model. In this case, the engine
        apply N points during the inference, following some methodology of set points into each seismic facie (The standard
        methodology implemented here is that of Kirillov, lead author of the SAM paper). Because the
        engine set N points during the inference, we get N predictions for each seismic facies. So we need return a list of
        dictionaries, that the key is the number of points used in prediction of each seismic facie and the value is the
        batch segmented. It is important to note that only model's forward are wrapped, and, thus, any method that requires
        the forward method (e.g., training_step, predict_step) will be performed using the engine, transparently to the user.

        Parameters
        ----------
        model : L.LightningModule
            Model (Sam), that will be used in inference.
        num_facies: int
            Number of classes (in seismic is called facies) that will be used
            for cut the image and predict each class, separately.
        multimask_output: bool
            Parameter used by Sam model for control if you would 1 or n-1 predictions.
            Sam model can predict only binary masks, so when we define num_facies
            we too define how many facies will be predicted. Each of this predictions
            represent a hypotesis of the object. If multimask_output=True, the model
            will return all of num_facies hypotesis, but if not, return the first prediction, beacause,
            in tesis, this mask is equal to all hypotesis (read paper for more details).
        num_points: int
            How many points used in each facie during prediction.

        """
        super().__init__()

        self.model = model
        self.num_points = num_points

        self.prompt_inferencer = PromptInferencerEngine(
            num_facies=num_facies,
            multimask_output=multimask_output,
            num_points=num_points,
        )

    def __call__(self, batched_input: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return self.forward(batched_input)

    def forward(self, batched_input: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Perform inference with points.

        Parameters
        ----------
        batched_input : List[Dict[str, Any]]
            Batch of input data. Need to be a list of dict (default of SAM).
        """
        return self.prompt_inferencer(self.model, batched_input=batched_input)

    def _single_step(
        self, batch: torch.Tensor, batch_idx: int, step_name: str
    ) -> List[Dict[str, Any]]:
        """Perform a single step of the training/validation loop.

        Parameters
        ----------
        batch : torch.Tensor
            The input data. This use SimpleDataset(), so batch is a tuple (x, y).
        batch_idx : int
            The index of the batch.
        step_name : str
            The name of the step, either "train" or "val".

        Returns
        -------
        torch.Tensor
            The predict value.
        """
        # making batch a list of dict for Sam() forward.
        batched_input = []
        for sample_idx in range(len(batch[0])):
            batched_input.append(
                {
                    "image": batch[0][sample_idx],
                    "label": batch[1][sample_idx],
                    "original_size": (
                        batch[0][sample_idx].shape[1],
                        batch[0][sample_idx].shape[2],
                    ),
                }
            )

        outputs = self.forward(batched_input)  # [B, num_points, H, W]

        return outputs

    def predict_step(self, batch: torch.Tensor, batch_idx: int):
        return self._single_step(batch, batch_idx, "predict")


class PromptInferencerEngine(_Engine):
    def __init__(
        self,
        num_facies: int = 6,
        multimask_output: bool = False,
        num_points: int = 10,
        custom_call: Optional[Callable] = None,
    ):
        """
        Class used to implement Kirillov's algorithm.

        Parameters
        ----------
        num_facies: int
            Number of classes (in seismic is called facies) that will be used
            for cut the image and predict each class, separately.
        multimask_output: bool
            Parameter used by Sam model for control if you would 1 or n-1 predictions.
            Sam model can predict only binary masks, so when we define num_facies
            we too define how many facies will be predicted. Each of this predictions
            represent a hypotesis of the object. If multimask_output=True, the model
            will return all of num_facies hypotesis, but if not, return the first prediction, beacause,
            in tesis, this mask is equal to all hypotesis (read paper for more details).
        num_points: int
            How many points used in each facie during prediction.
        custom_call: Callable, None
            This parameter is used when you need implement some new features,
            for exemple some new method based in main algorithm. Using this parameter
            you can implement __call__ method outside of this class and pass it here.
        """
        self.num_facies = num_facies
        self.multimask_output = multimask_output
        self.num_points = num_points
        self.custom_call = custom_call

        self.set_points()  # Reset accumulated points

    def set_points(self):
        """Reset the accumulated points"""
        self.accumulated_coords = np.empty((0, 2), dtype=int)  # Nx2 array
        self.accumulated_labels = np.empty((0,), dtype=int)  # Array of N length

    def __call__(self, model: Union[L.LightningModule, torch.nn.Module], batched_input: List[Dict[str, Any]]):  # type: ignore
        """
        Performs inference using facie-centered points and multiple binary thresholds.

        Each image in the batch is processed facie by facie, generating multiple inference points
        (up to `self.num_points`) per threshold. For each facie, computes IoU, intersection, union,
        prediction masks, and losses.

        PS: It is possible override this method using 'custom_call' parameter. If you need implement some new features,
        implement the method outside of class and pass it to this class. For example:
        ```
        def my_call_function(self, model, batched_input):
            pass
        engine = PromptInferencerEngine(custom_call=my_call_function)
        result = engine(model, batched_input)
        ```

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
        """
        if self.custom_call:
            return self.custom_call(self, model, batched_input)

        thresholds = [0.5]
        all_outputs = []

        for item in batched_input:
            image = item["image"]
            label = item["label"]
            original_size = item["original_size"]
            num_facies = torch.unique(label)  # identify the classes of sample
            facie_outputs_per_threshold = {}  # save predictions by threshold

            for threshold in thresholds:
                facie_outputs = {}
                for facie in num_facies:
                    region = (
                        (label == facie).to(torch.int32).to(model.device)
                    )  # define the region (binary label of seismic facie)
                    if (
                        torch.sum(region).item() <= 1
                    ):  # ignore false label (label that have 1 pixel)
                        continue

                    real_label = region  # default approach: we use real_label with label reference
                    self.set_points()
                    point_type = "positive"  # first point is positive
                    facie_outputs[int(facie)] = {
                        "ious": [],
                        "intersections": [],
                        "unions": [],
                        "points": [],
                        "mask_logits": [],
                        "label": [],
                        "losses": [],
                    }

                    prev_low_res_logits = None  # torch.zeros(1, 1, H, W, device=self.device) # used to send last mask to current prediction

                    for _ in range(self.num_points):
                        try:
                            point_coords, point_labels = self.calculate_center_region(
                                region.cpu().numpy(), point_type
                            )
                        except ValueError:
                            print("No regions found. Using previous points.")
                            point_coords, point_labels = (
                                self.accumulated_coords,
                                self.accumulated_labels,
                            )

                        # create the tensors from the points (necessary by model)
                        point_coords_tensor = torch.tensor(point_coords, dtype=torch.long).unsqueeze(0).to(model.device)  # type: ignore
                        point_labels_tensor = torch.tensor(point_labels, dtype=torch.long).unsqueeze(0).to(model.device)  # type: ignore

                        batch_dict = {
                            "image": image,
                            "label": real_label,
                            "original_size": original_size,
                            "point_coords": point_coords_tensor,
                            "point_labels": point_labels_tensor,
                        }

                        if prev_low_res_logits is not None:
                            batch_dict["mask_inputs"] = prev_low_res_logits

                        # execute prediction
                        # PS: this process of inference is executed by sample, so B (batch) aways be 1.
                        outputs = model(
                            [batch_dict], multimask_output=True
                        )  # default: self.multimask_output
                        iou_predictions = torch.stack(
                            [output["iou_predictions"].squeeze(0) for output in outputs]
                        )  # [B, 1]
                        low_res_logits = torch.stack(
                            [output["low_res_logits"].squeeze(0) for output in outputs]
                        )  # [B, 1, H, W]
                        masks_logits = torch.stack(
                            [output["masks_logits"].squeeze(0) for output in outputs]
                        )  # [B, 1, H, W]
                        labels = (real_label).unsqueeze(0)  # [B, H, W]

                        # calculate loss
                        loss_pred = model._loss(iou_predictions, masks_logits, labels)  # type: ignore

                        best_idx = model.best_idx  # best idx with min loss

                        prev_low_res_logits = (
                            low_res_logits[0, best_idx].unsqueeze(0).unsqueeze(0)
                        )

                        # apply sigmoid and treshold
                        sigmoid_masks = torch.sigmoid(
                            masks_logits[0, best_idx]
                        )  # [B, 1, H, W]
                        masks = sigmoid_masks > threshold  # [B, 1, H, W]

                        # calculate intersections, unions for micro-iou
                        pred_mask = masks.squeeze(1).bool()  # [B, H, W]
                        gt_mask = labels.bool()  # [B, H, W]
                        inter = int((pred_mask & gt_mask).sum().item())
                        uni = int((pred_mask | gt_mask).sum().item())
                        iou = float(inter / uni) if uni != 0 else 0.0

                        # get diff and update new point type
                        diff, new_point_type = self.calculate_diff_label_pred(
                            label=(real_label.cpu().numpy()),
                            pred=masks.squeeze().cpu().numpy(),
                        )
                        region = torch.tensor(diff, dtype=torch.int32).to(model.device)  # type: ignore
                        point_type = new_point_type

                        facie_outputs[int(facie)]["ious"].append(iou)
                        facie_outputs[int(facie)]["intersections"].append(inter)
                        facie_outputs[int(facie)]["unions"].append(uni)
                        facie_outputs[int(facie)]["points"].append(
                            (
                                point_coords_tensor.cpu().numpy(),
                                point_labels_tensor.cpu().numpy(),
                            )
                        )
                        facie_outputs[int(facie)]["mask_logits"].append(
                            masks_logits[0, best_idx].cpu().numpy()
                        )
                        facie_outputs[int(facie)]["label"].append(labels.cpu().numpy())
                        facie_outputs[int(facie)]["losses"].append(loss_pred.item())

                facie_outputs_per_threshold[threshold] = facie_outputs
            all_outputs.append(facie_outputs_per_threshold)
        return all_outputs

    # """ version 1 (calculate the center of region, interativaly with model) """
    def calculate_center_region(self, region, point_type: str, min_distance: int = 10):
        """
        Calculates the centroid of the largest region of white pixels in a binary image,
        ensuring that the point is in the vertical center of the white region.

        Args:
            region (torch tensor):
                Binary image with region of interest (white pixels).
            point_type (str):
                Point type ('positive' or 'negative').
            min_distance (int):
                Minimum distance allowed between points.

        Returns:
            point_coords (np.ndarray): Nx2 array of accumulated points.
            point_labels (np.ndarray): Array N of accumulated labels.
        """
        # convert to numpy
        if isinstance(region, torch.Tensor):
            region = region.detach().cpu().numpy()
        else:
            region = region

        if not isinstance(region, np.ndarray):
            raise TypeError("region needs to be a NumPy array.")

        region = region.astype(np.uint8)

        # find the connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(region, connectivity=8)  # type: ignore

        if num_labels < 2:  # only background and no white region
            raise ValueError("No connected white regions found in the binary image.")

        # ignore label 0 (bottom), take the largest connected component
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])  # type: ignore
        mask_largest = labels == largest_label  # mask of the largest white region

        # find the white coordinates within the largest region
        y_indices, x_indices = np.where(mask_largest)

        if len(y_indices) == 0:
            raise ValueError(
                "The largest connected component contains no white pixels."
            )

        # find the central column of the largest white region
        center_x = np.median(x_indices).astype(int)  # center point at X

        # finding the actual vertical center
        y_in_column = y_indices[x_indices == center_x]
        if len(y_in_column) > 0:
            center_y = y_in_column[
                len(y_in_column) // 2
            ]  # get a real value at the center of the vertical strip
        else:
            raise ValueError("center_y variable not defined.")

        new_coords = np.array([[center_x, center_y]])

        # check if the point is too close to the previous ones
        if self.accumulated_coords.shape[0] > 0:
            distances = np.sqrt(
                np.sum((self.accumulated_coords - new_coords) ** 2, axis=1)
            )
            if np.any(distances < min_distance):

                # try to move the point horizontally within the white region
                region_height, region_width = region.shape
                for delta_x in range(
                    min_distance, region_width, min_distance
                ):  # increments in min_distance
                    candidate_x_right = center_x + delta_x
                    candidate_x_left = center_x - delta_x

                    # check right first, then left
                    if (
                        candidate_x_right < region_width
                        and region[center_y, candidate_x_right] > 0
                    ):
                        center_x = candidate_x_right
                        break
                    elif (
                        candidate_x_left >= 0 and region[center_y, candidate_x_left] > 0
                    ):
                        center_x = candidate_x_left
                        break

                new_coords = np.array([[center_x, center_y]])

        # define labels (positive or negative)
        if point_type == "positive":
            new_labels = np.array([1])
        elif point_type == "negative":
            new_labels = np.array([0])
        else:
            raise ValueError("Invalid point_type. Must be 'positive' or 'negative'.")

        # accumulate the results
        self.accumulated_coords = np.vstack([self.accumulated_coords, new_coords])
        self.accumulated_labels = np.hstack([self.accumulated_labels, new_labels])

        return self.accumulated_coords, self.accumulated_labels

    def calculate_diff_label_pred(self, label: np.ndarray, pred: np.ndarray):
        """
        Calculates the difference between two binary images and determines whether the outer or inner area is larger.

        Args:
            label (np.ndarray):
                Reference binary image (label).
            pred (np.ndarray):
                Predicted binary image (pred).

        Returns:
            diff_colored (np.ndarray): Color image representing the differences.
            point_type (str): 'negative' if the external area is larger, 'positive' if the internal area is larger.
        """
        if label.shape != pred.shape:
            raise ValueError(
                "Label and Pred images have differents shapes. Check it before call calculate_dif_label_pred() function."
            )

        # masks for difference regions
        mask_outward = label > pred  # Difference out -> Red
        mask_inward = label < pred  # Difference inside -> Blue

        area_outward = np.sum(mask_outward)
        area_inward = np.sum(mask_inward)

        diff_binary = np.zeros(label.shape, dtype=np.int32)  # [H,W]

        # compare the areas
        if area_outward > area_inward:
            diff_binary[mask_outward] = 1
            point_type = "positive"
        else:
            diff_binary[mask_inward] = 1
            point_type = "negative"

        return diff_binary, point_type

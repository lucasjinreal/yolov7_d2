import importlib
from pathlib import Path
from typing import Union, Dict, Any

import cv2
import numpy as np
import wandb
from alfred.vis.image.get_dataset_label_map import coco_label_map_list
from detectron2.utils.visualizer import GenericMask

coco_label_map = {k: v for k, v in enumerate(coco_label_map_list) if isinstance(v, str)}


def is_wandb_available():
    return importlib.util.find_spec("wandb") is not None


class WandbFormatter:
    """Converts detectron2 output to wandb.Image arguments"""

    def __init__(
        self,
        image_path: Union[str, Path],
        class_names: Dict[Any, Any],
        conf_threshold: float = 0.7,
    ):
        self.image_path = image_path
        self.image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        self.class_names = class_names
        self.conf_threshold = conf_threshold
        self.class_set = wandb.Classes(
            [{"id": idx, "name": name} for idx, name in self.class_names.items()]
        )

    def convert_instance_predictions(self, predictions):
        """
        Converts instance-level prediction results for an image.

        Args:
            predictions (Instances): the output of an instance detection/segmentation
                model. Following fields will be used to create the final dictionary to pass to wandb
                "pred_boxes", "pred_classes", "scores", "pred_masks".

        Returns:
            output Dict[str,Any]: image with kwargs for wandb logger
        """
        boxes = (
            predictions.pred_boxes.tensor.cpu().numpy().tolist()
            if predictions.has("pred_boxes")
            else None
        )
        scores = (
            predictions.scores.cpu().numpy().tolist()
            if predictions.has("scores")
            else None
        )
        classes = (
            predictions.pred_classes.cpu().numpy().tolist()
            if predictions.has("pred_classes")
            else None
        )

        if predictions.has("pred_masks"):
            masks = predictions.pred_masks.cpu().numpy()
            masks = [
                GenericMask(x, self.image.shape[0], self.image.shape[1]) for x in masks
            ]
        else:
            masks = None

        num_objects = 0
        confidences = []

        if masks is not None:
            boxes = []
            final_mask = np.zeros(
                (self.image.shape[0], self.image.shape[1]), dtype=np.uint8
            )
            for i, mask in enumerate(masks):
                if scores[i] > self.conf_threshold:
                    pred_mask = mask.mask
                    try:
                        boxes.append(mask.bbox())
                    except IndexError:
                        pass
                    pred_class = int(classes[i]) + 1
                    final_mask = np.ma.array(final_mask, mask=pred_mask)
                    final_mask = final_mask.filled(pred_class)
                    num_objects += 1
                    confidences.append(scores[i])
                final_mask = final_mask.astype(np.uint8)
                masks = {
                    "prediction": {
                        "mask_data": final_mask,
                        "class_labels": self.class_names,
                    }
                }
        if boxes is not None:
            boxes_data = []
            for i, box in enumerate(boxes):
                if scores[i] > self.conf_threshold:
                    pred_class = int(classes[i]) + 1
                    caption = (
                        f"{pred_class}"
                        if not self.class_names
                        else self.class_names[pred_class]
                    )
                    boxes_data.append(
                        {
                            "position": {
                                "minX": box[0],
                                "minY": box[1],
                                "maxX": box[2],
                                "maxY": box[3],
                            },
                            "class_id": pred_class,
                            "box_caption": f"{i}: {caption} @ {scores[i] * 100:.2f}%",
                            "scores": {"class_score": scores[i]},
                            "domain": "pixel",
                        }
                    )
                    if masks is None:
                        confidences.append(scores[i])
                        num_objects += 1
            if boxes_data:
                boxes = {
                    "prediction": {
                        "box_data": boxes_data,
                        "class_labels": self.class_names,
                    }
                }
            else:
                boxes = None
        row = (
            str(Path(self.image_path).name),
            wandb.Image(
                data_or_path=self.image_path,
                boxes=boxes,
                masks=masks,
                classes=self.class_set,
            ),
            num_objects,
            confidences,
        )
        return row


class WandbInferenceLogger:
    """
    Logs inference images and predictions to wandb.
    Currently, supports bounding boxes and instance segmentation.
    """

    def __init__(
        self,
        wandb_entity: str = None,
        wandb_project: str = None,
        class_names: Dict[int, str] = None,
        conf_threshold: float = 0.7,
        config=None,
    ):
        if not is_wandb_available():
            raise ImportError(
                "Please install wandb using 'pip install wandb --upgrade'"
            )

        self.class_names = class_names if class_names else coco_label_map
        self.wandb = wandb
        self.run = None
        if wandb.run:
            self.run = wandb.run
        else:
            if wandb_project is None:
                raise ValueError("wandb_project is required for wandb logger ")
            self.run = wandb.init(
                project=wandb_project, entity=wandb_entity, config=config,
            )
        self.dataset_name = self.run.id + "_dataset"
        self.conf_threshold = conf_threshold
        self.table: wandb.Table = self.wandb.Table(
            columns=[
                "Image-File",
                "Predictions",
                "Number-of-Objects",
                "Prediction-Confidence",
            ]
        )

    def log_inference(self, image, result):
        """adds the inference result to a table in wandb."""
        if not self.run:
            return None
        formatter = WandbFormatter(
            image, class_names=self.class_names, conf_threshold=self.conf_threshold
        )
        image_name = str(Path(image).stem)
        instance_prediction = formatter.convert_instance_predictions(
            result["instances"]
        )
        self.table.add_data(*instance_prediction)

    def finish_run(self):
        """Uploads the table to wandb, finishes the run."""
        if not self.run:
            return None
        self.run.log({self.dataset_name: self.table})
        self.run.finish()

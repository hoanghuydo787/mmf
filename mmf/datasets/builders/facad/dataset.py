import copy
import json

import torch
from mmf.common.sample import Sample
from mmf.common.typings import MMFDatasetConfigType
from mmf.datasets.mmf_dataset import MMFDataset
from .database import FACADDatabase


class FACADDataset(MMFDataset):
    def __init__(
        self,
        config: MMFDatasetConfigType,
        dataset_type: str,
        index: int,
        *args,
        **kwargs,
    ):
        super().__init__(
            "facad",
            config,
            dataset_type,
            index,
            FACADDatabase,
            *args,
            **kwargs,
        )

    def init_processors(self):
        super().init_processors()
        # Assign transforms to the image_db
        if self._dataset_type == "train":
            self.image_db.transform = self.train_image_processor
        else:
            self.image_db.transform = self.eval_image_processor

    def _get_valid_text_attribute(self, sample_info):
        if "captions" in sample_info:
            return "captions"

        if "sentences" in sample_info:
            return "sentences"

        raise AttributeError("No valid text attribution was found")

    def __getitem__(self, idx):
        sample_info = self.annotation_db[idx]
        text_attr = self._get_valid_text_attribute(sample_info)

        current_sample = Sample()
        sentence = sample_info[text_attr]
        processed_sentence = self.text_processor({"text": sentence})

        current_sample.text = processed_sentence["text"]
        if "input_ids" in processed_sentence:
            current_sample.update(processed_sentence)
        current_sample.image = self.image_db[idx]["images"][0]
        current_sample.ann_idx = torch.tensor(idx, dtype=torch.long)
        current_sample.targets = None  # Dummy for Loss

        return current_sample

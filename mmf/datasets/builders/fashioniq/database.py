import os
import json

import pandas as pd
import torch
from mmf.utils.file_io import PathManager


class FashionIQDatabase(torch.utils.data.Dataset):
    SPLITS = {"train": "train", "val": "val", "test": "val"}

    def __init__(self, config, splits_path, dataset_type, *args, **kwargs):
        super().__init__()
        self.garment_class = config.get("garment_class", "all")
        assert self.garment_class in ["all", "shirt", "dress", "toptee"]
        self.dataset_type = dataset_type
        self.splits = self.SPLITS[self.dataset_type]
        self._load_annotation_db(splits_path)

    def _load_annotation_db(self, splits_path):
        data = []

        if self.garment_class == "all":
            annotations_json = []
            for garment_class in ["shirt", "dress", "toptee"]:
                json_name = "cap.{}.{}.json".format(garment_class, self.splits)
                with PathManager.open(os.path.join(splits_path, json_name), "r") as f:
                    annotations_json.extend(json.load(f))
        else:
            json_name = "cap.{}.{}.json".format(self.garment_class, self.splits)
            with PathManager.open(os.path.join(splits_path, json_name), "r") as f:
                annotations_json = json.load(f)

        for item in annotations_json:
            data.append(
                {
                    "ref_path": item["candidate"] + ".jpg",
                    "tar_path": item["target"] + ".jpg",
                    "sentences": ", ".join(item["captions"]),
                }
            )

        if len(data) == 0:
            raise RuntimeError("Dataset is empty")

        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
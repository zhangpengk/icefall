
import argparse
import inspect
import logging

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from lhotse import (
    CutSet,
    Fbank,
    FbankConfig,
    load_manifest_lazy,
)

from lhotse.dataset import (
    CutConcatenate,
    DynamicBucketingSampler,
    K2SpeechRecognitionDataset,
    PrecomputedFeatures,
    SimpleCutSampler,
    SpecAugment,
)

from lhotse.dataset.input_strategies import OnTheFlyFeatures
from lhotse.utils import fix_random_seed

from torch.utils.data import DataLoader

from icefall.utils import str2bool


class _SeedWorkers:
    def __init__(self, seed: int):
        self.seed = seed

    def __call__(self, worker_id: int):
        fix_random_seed(self.seed + worker_id)


class WenetSpeechAsrDataModule:
    """
    DataModule for THCHS-30 phoneme-based Zipformer training.

    保留 WenetSpeech KWS Recipe 的类名和主要接口，
    但数据入口改为 THCHS-30。

    Manifest naming convention:

        thchs_30_feats_cuts_train.jsonl.gz
        thchs_30_feats_cuts_dev.jsonl.gz
        thchs_30_feats_cuts_test.jsonl.gz

    Also supports sharded manifests, e.g.:

        thchs_30_feats_cuts_train.1.jsonl.gz
        thchs_30_feats_cuts_train.2.jsonl.gz
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):

        group = parser.add_argument_group(
            title="THCHS-30 data related options",
            description="DataLoader and dataset options.",
        )

        group.add_argument(
            "--manifest-dir",
            type=Path,
            default=Path("data/fbank"),
            help="Directory containing THCHS-30 manifests.",
        )

        group.add_argument(
            "--max-duration",
            type=float,
            default=100.0,
            help="Maximum duration in a batch.",
        )

        group.add_argument(
            "--bucketing-sampler",
            type=str2bool,
            default=True,
            help="Use DynamicBucketingSampler.",
        )

        group.add_argument(
            "--num-buckets",
            type=int,
            default=10,
            help="Number of buckets.",
        )

        group.add_argument(
            "--concatenate-cuts",
            type=str2bool,
            default=False,
            help="Whether to concatenate cuts.",
        )

        group.add_argument(
            "--duration-factor",
            type=float,
            default=1.0,
            help="Duration factor for concatenation.",
        )

        group.add_argument(
            "--gap",
            type=float,
            default=1.0,
            help="Gap between concatenated cuts.",
        )

        group.add_argument(
            "--on-the-fly-feats",
            type=str2bool,
            default=False,
            help="Extract features online instead of using precomputed features.",
        )

        group.add_argument(
            "--shuffle",
            type=str2bool,
            default=True,
            help="Shuffle training cuts.",
        )

        group.add_argument(
            "--return-cuts",
            type=str2bool,
            default=True,
            help="Return original cuts in each batch.",
        )

        group.add_argument(
            "--num-workers",
            type=int,
            default=2,
            help="Number of DataLoader workers.",
        )

        group.add_argument(
            "--enable-spec-aug",
            type=str2bool,
            default=True,
            help="Enable SpecAugment.",
        )

        group.add_argument(
            "--spec-aug-time-warp-factor",
            type=int,
            default=80,
            help="SpecAugment time warp factor.",
        )

        # 默认不加载 MUSAN。
        # 这样数据管线只依赖 THCHS-30 manifest。
        # 如果需要噪声增强，可以后续单独增加。

    # ---------------------------------------------------------
    # Manifest loading
    # ---------------------------------------------------------

    def _find_manifests(self, split: str) -> List[Path]:
        """
        Find THCHS-30 manifests for a specific split.

        split:
            train / dev / test
        """

        manifest_dir = self.args.manifest_dir

        if not manifest_dir.exists():
            raise FileNotFoundError(
                f"Manifest directory does not exist: {manifest_dir}"
            )

        all_paths = sorted(
            manifest_dir.glob("thchs_30_cuts*.jsonl")
        )

        if not all_paths:
            raise FileNotFoundError(
                f"No THCHS-30 manifests found in {manifest_dir}"
            )

        result = []

        for path in all_paths:
            name = path.name.lower()

            # 匹配 split，避免 train 和 test 混淆
            if split == "train":
                matched = (
                    "_train" in name
                    or "_training" in name
                )

            elif split == "dev":
                matched = (
                    "_dev" in name
                    or "_valid" in name
                    or "_validation" in name
                )

            elif split == "test":
                matched = "_test" in name

            else:
                raise ValueError(f"Unsupported split: {split}")

            if matched:
                result.append(path)

        if not result:
            raise FileNotFoundError(
                f"No {split} manifest found. "
                f"Available files: {[p.name for p in all_paths]}"
            )

        logging.info(
            f"Found {len(result)} THCHS-30 {split} manifests:"
        )

        for path in result:
            logging.info(f"  {path}")

        return result

    def _load_split(self, split: str) -> CutSet:
        """
        Load one split. Multiple manifest shards are concatenated.
        """

        paths = self._find_manifests(split)

        cuts = load_manifest_lazy(paths[0])

        for path in paths[1:]:
            cuts = cuts + load_manifest_lazy(path)

        return cuts

    @lru_cache()
    def train_cuts(self) -> CutSet:
        logging.info("Loading THCHS-30 training cuts")
        return self._load_split("train")

    @lru_cache()
    def valid_cuts(self) -> CutSet:
        logging.info("Loading THCHS-30 validation cuts")
        return self._load_split("dev")

    @lru_cache()
    def test_cuts(self) -> CutSet:
        logging.info("Loading THCHS-30 test cuts")
        return self._load_split("test")

    # ---------------------------------------------------------
    # Training DataLoader
    # ---------------------------------------------------------

    def train_dataloaders(
        self,
        cuts_train: CutSet,
        sampler_state_dict: Optional[Dict[str, Any]] = None,
    ) -> DataLoader:

        transforms = []

        if self.args.concatenate_cuts:
            logging.info(
                "Enable CutConcatenate: "
                f"duration_factor={self.args.duration_factor}, "
                f"gap={self.args.gap}"
            )

            transforms.append(
                CutConcatenate(
                    duration_factor=self.args.duration_factor,
                    gap=self.args.gap,
                )
            )

        input_transforms = []

        if self.args.enable_spec_aug:
            logging.info("Enable SpecAugment")

            # 兼容不同版本 Lhotse
            num_frame_masks = 10

            num_frame_masks_parameter = inspect.signature(
                SpecAugment.__init__
            ).parameters["num_frame_masks"]

            if num_frame_masks_parameter.default == 1:
                num_frame_masks = 2

            input_transforms.append(
                SpecAugment(
                    time_warp_factor=self.args.spec_aug_time_warp_factor,
                    num_frame_masks=num_frame_masks,
                    features_mask_size=27,
                    num_feature_masks=2,
                    frames_mask_size=100,
                )
            )

        else:
            logging.info("Disable SpecAugment")

        logging.info("Creating THCHS-30 training dataset")

        train = K2SpeechRecognitionDataset(
            cut_transforms=transforms,
            input_transforms=input_transforms,
            input_strategy=PrecomputedFeatures(),
            return_cuts=self.args.return_cuts,
        )

        if self.args.on_the_fly_feats:
            logging.info("Using OnTheFlyFeatures")

            train = K2SpeechRecognitionDataset(
                cut_transforms=transforms,
                input_transforms=input_transforms,
                input_strategy=OnTheFlyFeatures(
                    Fbank(
                        FbankConfig(
                            num_mel_bins=80
                        )
                    )
                ),
                return_cuts=self.args.return_cuts,
            )

        # Sampler
        if self.args.bucketing_sampler:
            logging.info("Using DynamicBucketingSampler")

            train_sampler = DynamicBucketingSampler(
                cuts_train,
                max_duration=self.args.max_duration,
                shuffle=self.args.shuffle,
                num_buckets=self.args.num_buckets,
                buffer_size=self.args.num_buckets * 2000,
                shuffle_buffer_size=self.args.num_buckets * 5000,
                drop_last=True,
            )

        else:
            logging.info("Using SimpleCutSampler")

            train_sampler = SimpleCutSampler(
                cuts_train,
                max_duration=self.args.max_duration,
                shuffle=self.args.shuffle,
            )

        if sampler_state_dict is not None:
            logging.info("Loading sampler state dict")
            train_sampler.load_state_dict(sampler_state_dict)

        seed = torch.randint(0, 100000, ()).item()
        worker_init_fn = _SeedWorkers(seed)

        train_dl = DataLoader(
            train,
            sampler=train_sampler,
            batch_size=None,
            num_workers=self.args.num_workers,
            persistent_workers=False,
            worker_init_fn=worker_init_fn,
        )

        return train_dl

    # ---------------------------------------------------------
    # Validation DataLoader
    # ---------------------------------------------------------

    def valid_dataloaders(
        self,
        cuts_valid: CutSet,
    ) -> DataLoader:

        transforms = []

        if self.args.concatenate_cuts:
            transforms.append(
                CutConcatenate(
                    duration_factor=self.args.duration_factor,
                    gap=self.args.gap,
                )
            )

        logging.info("Creating THCHS-30 validation dataset")

        if self.args.on_the_fly_feats:
            validate = K2SpeechRecognitionDataset(
                cut_transforms=transforms,
                input_strategy=OnTheFlyFeatures(
                    Fbank(
                        FbankConfig(
                            num_mel_bins=80
                        )
                    )
                ),
                return_cuts=self.args.return_cuts,
            )

        else:
            validate = K2SpeechRecognitionDataset(
                cut_transforms=transforms,
                input_strategy=PrecomputedFeatures(),
                return_cuts=self.args.return_cuts,
            )

        valid_sampler = DynamicBucketingSampler(
            cuts_valid,
            max_duration=self.args.max_duration,
            shuffle=False,
        )

        valid_dl = DataLoader(
            validate,
            sampler=valid_sampler,
            batch_size=None,
            num_workers=self.args.num_workers,
            persistent_workers=False,
        )

        return valid_dl

    # ---------------------------------------------------------
    # Test DataLoader
    # ---------------------------------------------------------

    def test_dataloaders(
        self,
        cuts: CutSet,
    ) -> DataLoader:

        logging.info("Creating THCHS-30 test dataset")

        if self.args.on_the_fly_feats:
            input_strategy = OnTheFlyFeatures(
                Fbank(
                    FbankConfig(
                        num_mel_bins=80
                    )
                )
            )
        else:
            input_strategy = PrecomputedFeatures()

        test = K2SpeechRecognitionDataset(
            input_strategy=input_strategy,
            return_cuts=self.args.return_cuts,
        )

        test_sampler = DynamicBucketingSampler(
            cuts,
            max_duration=self.args.max_duration,
            shuffle=False,
        )

        test_dl = DataLoader(
            test,
            batch_size=None,
            sampler=test_sampler,
            num_workers=self.args.num_workers,
        )

        return test_dl

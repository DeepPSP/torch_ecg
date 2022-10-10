"""
"""

import json
import warnings
from copy import deepcopy
from dataclasses import dataclass
from random import shuffle
from typing import Dict, List, Sequence, Optional, Union, NoReturn

import numpy as np

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Config, BatchFeature
from transformers.pytorch_utils import torch_int_div
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    _compute_mask_indices,
    _sample_negative_indices,
)
from torch_ecg.cfg import CFG
from torch_ecg.utils.misc import ReprMixin, list_sum
from torch_ecg.utils.utils_data import stratified_train_test_split
from torch_ecg._preprocessors import PreprocManager

from .pretraining_cfg import PreTrainModelCfg
from data_reader import CINC2022Reader, CINC2016Reader, EPHNOGRAMReader, CompositeReader


__all__ = [
    "DataCollatorForWav2Vec2Pretraining",
    "get_pretraining_datacollator",
    "Wav2Vec2PretrainingDataset",
]


@dataclass
class DataCollatorForWav2Vec2Pretraining:
    """
    Data collator that will dynamically pad the inputs received and prepare masked indices
    for self-supervised pretraining.
    Args:
        model (:class:`~transformers.Wav2Vec2ForPreTraining`):
            The Wav2Vec2 model used for pretraining. The data collator needs to have access
            to config and ``_get_feat_extract_output_lengths`` function for correct padding.
        feature_extractor (:class:`~transformers.Wav2Vec2FeatureExtractor`):
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    config: Wav2Vec2Config
    feature_extractor: Wav2Vec2FeatureExtractor
    padding: Union[bool, str] = "longest"
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        """
        reformat list to dict and set to pytorch format

        Parameters
        ----------
        features : List[Dict[str, Union[List[int], torch.Tensor]]]
            List of features to be processed.
            features are dicts with the following keys:
                - "input_values": List (Tensor) of input values.
                - "attention_mask": Optional (List or Tensor) of attention mask.

        """
        batch = self.feature_extractor.pad(
            features,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        device = batch["input_values"].device
        batch_size = batch["input_values"].shape[0]

        mask_indices_seq_length = self._get_feat_extract_output_lengths(
            batch["input_values"].shape[-1]
        )
        # make sure masked sequence length is a Python scalar
        mask_indices_seq_length = int(mask_indices_seq_length)

        # make sure that no loss is computed on padded inputs
        if batch.get("attention_mask") is not None:
            # compute real output lengths according to convolution formula
            batch["sub_attention_mask"] = self._get_feature_vector_attention_mask(
                mask_indices_seq_length, batch["attention_mask"]
            )

        features_shape = (batch_size, mask_indices_seq_length)

        # sample randomly masked indices
        mask_time_indices = _compute_mask_indices(
            features_shape,
            self.config.mask_time_prob,
            self.config.mask_time_length,
            attention_mask=batch.get("sub_attention_mask"),
        )

        # sample negative indices
        sampled_negative_indices = _sample_negative_indices(
            features_shape,
            self.config.num_negatives,
            mask_time_indices=mask_time_indices,
        )
        batch["mask_time_indices"] = torch.tensor(
            mask_time_indices, dtype=torch.long, device=device
        )
        batch["sampled_negative_indices"] = torch.tensor(
            sampled_negative_indices, dtype=torch.long, device=device
        )

        return batch

    def _get_feature_vector_attention_mask(
        self,
        feature_vector_length: int,
        attention_mask: torch.LongTensor,
        add_adapter: Optional[bool] = None,
    ) -> torch.LongTensor:
        # Effectively attention_mask.sum(-1), but not inplace to be able to run
        # on inference mode.
        non_padded_lengths = attention_mask.cumsum(dim=-1)[:, -1]

        output_lengths = self._get_feat_extract_output_lengths(
            non_padded_lengths, add_adapter=add_adapter
        )
        output_lengths = output_lengths.to(torch.long)

        batch_size = attention_mask.shape[0]

        attention_mask = torch.zeros(
            (batch_size, feature_vector_length),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        # these two operations makes sure that all values before the output lengths idxs are attended to
        attention_mask[
            (
                torch.arange(attention_mask.shape[0], device=attention_mask.device),
                output_lengths - 1,
            )
        ] = 1
        attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool()
        return attention_mask

    def _get_feat_extract_output_lengths(
        self,
        input_lengths: Union[torch.LongTensor, int],
        add_adapter: Optional[bool] = None,
    ) -> int:
        """
        Computes the output length of the convolutional layers
        """

        add_adapter = self.config.add_adapter if add_adapter is None else add_adapter

        def _conv_out_length(input_length: int, kernel_size: int, stride: int) -> int:
            # 1D convolutional layer output length formula taken
            # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            return torch_int_div(input_length - kernel_size, stride) + 1

        for kernel_size, stride in zip(
            self.config.conv_kernel, self.config.conv_stride
        ):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        if add_adapter:
            for _ in range(self.config.num_adapter_layers):
                input_lengths = _conv_out_length(
                    input_lengths, 1, self.config.adapter_stride
                )

        return input_lengths


def get_pretraining_datacollator(
    cfg: Optional[CFG] = None,
) -> DataCollatorForWav2Vec2Pretraining:
    """ """
    if cfg is None:
        cfg = PreTrainModelCfg
    assert hasattr(cfg, "get_Wav2Vec2Config"), "cfg must have get_Wav2Vec2Config method"
    assert hasattr(
        cfg, "get_Wav2Vec2FeatureExtractor"
    ), "cfg must have get_Wav2Vec2FeatureExtractor method"

    extra_options = {
        "padding": cfg.get("padding", None),
        "max_length": cfg.get("max_length", None),
        "pad_to_multiple_of": cfg.get("pad_to_multiple_of", None),
    }
    extra_options = {k: v for k, v in extra_options.items() if v is not None}
    return DataCollatorForWav2Vec2Pretraining(
        cfg.get_Wav2Vec2Config(),
        cfg.get_Wav2Vec2FeatureExtractor(),
        **extra_options,
    )


class Wav2Vec2PretrainingDataset(Dataset, ReprMixin):
    """ """

    __name__ = "Wav2Vec2PretrainingDataset"

    def __init__(
        self,
        config: CFG,
        feature_extractor: Wav2Vec2FeatureExtractor,
        training: bool = True,
        lazy: bool = True,
    ) -> NoReturn:
        """ """
        self.config = deepcopy(config)
        self.feature_extractor = feature_extractor
        self.training = training
        self.lazy = lazy

        data_readers = []
        self.records = []
        if self.config.get("cinc2022_dir", None) is not None:
            data_readers.append(CINC2022Reader(db_dir=self.config.cinc2022_dir))
            record_subset = self._train_test_split_cinc2022(data_readers[-1])
            self.records.append(record_subset)
        if self.config.get("cinc2016_dir", None) is not None:
            data_readers.append(CINC2016Reader(db_dir=self.config.cinc2016_dir))
            record_subset = self._train_test_split_cinc2016(data_readers[-1])
            self.records.append(record_subset)
        if self.config.get("ephnogram_dir", None) is not None:
            data_readers.append(EPHNOGRAMReader(db_dir=self.config.ephnogram_dir))
            record_subset = self._train_test_split_ephnogram(data_readers[-1])
            self.records.append(record_subset)
        assert len(data_readers) > 0 and len(self.records), "No training data!"

        self.reader = CompositeReader(data_readers, fs=self.config.fs)
        self.records = [
            self.reader.get_composite_record_name(dr, rec)
            for dr, l_rec in zip(data_readers, self.records)
            for rec in l_rec
        ]

        if self.training:
            shuffle(self.records)

        if self.config.torch_dtype == torch.float64:
            self.dtype = np.float64
        else:
            self.dtype = np.float32

        ppm_config = CFG(random=False)
        ppm_config.update(deepcopy(self.config))
        self.ppm = PreprocManager.from_config(ppm_config)

        self.fdr = FastDataReader(
            self.reader, self.records, self.config, self.feature_extractor, self.ppm
        )

        self._signals = None
        if not self.lazy:
            self._load_all_data()

    def __len__(self) -> int:
        """ """
        if self._signals is None:
            return 0
        return len(self._signals)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """ """
        if self.signals is None:
            # self._load_all_data()
            raise Exception("call _load_all_data() before iterating over the dataset")
        return self.signals[index]

    def _load_all_data(self) -> NoReturn:
        """ """
        if self._signals is not None and len(self._signals) > 0:
            return
        self._signals = []
        with tqdm(range(len(self.fdr)), desc="Loading data", unit="records") as pbar:
            for idx in pbar:
                self._signals.extend(self.fdr[idx])

    def _train_test_split_cinc2022(
        self,
        reader: CINC2022Reader,
        train_ratio: float = 0.8,
        force_recompute: bool = False,
    ) -> List[str]:
        """ """
        _train_ratio = int(train_ratio * 100)
        _test_ratio = 100 - _train_ratio
        assert _train_ratio * _test_ratio > 0

        train_file = reader.db_dir / f"train_ratio_{_train_ratio}.json"
        test_file = reader.db_dir / f"test_ratio_{_test_ratio}.json"

        if not force_recompute and train_file.exists() and test_file.exists():
            if self.training:
                train_subjects = json.loads(train_file.read_text())
            else:
                test_subjects = json.loads(test_file.read_text())
        else:
            df_train, df_test = stratified_train_test_split(
                reader.df_stats,
                [
                    "Murmur",
                    "Age",
                    "Sex",
                    "Pregnancy status",
                ],
                test_ratio=1 - train_ratio,
            )
            train_subjects = df_train["Patient ID"].tolist()
            test_subjects = df_test["Patient ID"].tolist()

            train_file.write_text(json.dumps(train_subjects, ensure_ascii=False))
            test_file.write_text(json.dumps(test_subjects, ensure_ascii=False))

        if self.training:
            subjects = train_subjects
        else:
            subjects = test_subjects

        df = reader.df_stats[reader.df_stats["Patient ID"].isin(subjects)]
        records = list_sum(
            [reader.subject_records[row["Patient ID"]] for _, row in df.iterrows()]
        )

        if self.training:
            shuffle(records)

        return records

    def _train_test_split_cinc2016(self, reader: CINC2016Reader, **kwargs) -> List[str]:
        """ """
        if kwargs:
            warnings.warn(
                "CinC2016 has officially prefixed validation set, keyword arguments are ignored."
            )
        if self.training:
            records = [rec for rec in reader if rec not in reader.validation_set]
            shuffle(records)
        else:
            records = reader.validation_set
        return records

    def _train_test_split_ephnogram(
        self,
        reader: EPHNOGRAMReader,
        train_ratio: float = 0.8,
        force_recompute: bool = False,
    ) -> List[str]:
        """ """
        _train_ratio = int(train_ratio * 100)
        _test_ratio = 100 - _train_ratio
        assert _train_ratio * _test_ratio > 0

        train_file = reader.db_dir / f"train_ratio_{_train_ratio}.json"
        test_file = reader.db_dir / f"test_ratio_{_test_ratio}.json"

        if not force_recompute and train_file.exists() and test_file.exists():
            if self.training:
                return json.loads(train_file.read_text())
            else:
                return json.loads(test_file.read_text())
        else:
            # TODO: should one split by subject or not?
            df = reader.df_stats.drop_duplicates(subset=["Subject ID"])
            df_train, df_test = stratified_train_test_split(
                df,
                [
                    # "Age (years)", shoule be categorized before used for stratified split
                    "Gender",
                ],
                test_ratio=1 - train_ratio,
            )
            train_set = reader.df_stats[
                reader.df_stats["Subject ID"].isin(df_train["Subject ID"])
            ]["Record Name"].tolist()
            test_set = reader.df_stats[
                reader.df_stats["Subject ID"].isin(df_test["Subject ID"])
            ]["Record Name"].tolist()

            train_file.write_text(json.dumps(train_set, ensure_ascii=False))
            test_file.write_text(json.dumps(test_set, ensure_ascii=False))

        if self.training:
            shuffle(train_set)
            return train_set
        else:
            return test_set

    @property
    def signals(self) -> np.ndarray:
        return self._signals

    def extra_repr_keys(self) -> List[str]:
        """ """
        return ["training", "feature_extractor", "config"]


class FastDataReader(Dataset, ReprMixin):
    """ """

    def __init__(
        self,
        reader: CompositeReader,
        records: Sequence[str],
        config: CFG,
        feature_extractor: Wav2Vec2FeatureExtractor,
        ppm: Optional[PreprocManager] = None,
    ) -> NoReturn:
        """ """
        self.reader = reader
        self.records = records
        self.config = config
        self.feature_extractor = feature_extractor
        self.ppm = ppm
        if self.config.torch_dtype == torch.float64:
            self.dtype = np.float64
        else:
            self.dtype = np.float32
        self.max_length = int(self.config.max_duration_in_seconds * self.config.fs)
        if self.config.min_duration_in_seconds is not None:
            self.min_length = int(self.config.min_duration_in_seconds * self.config.fs)
        else:
            self.min_length = 0

    def __len__(self) -> int:
        """ """
        return len(self.records)

    def __getitem__(self, index: int) -> List[BatchFeature]:
        """ """
        rec = self.records[index]
        values = self.reader.load_data(rec, data_format="channel_first")
        if self.ppm:
            values, _ = self.ppm(values, self.reader.fs)
        if values.shape[-1] > self.max_length:
            n_segments, res = divmod(values.shape[-1], self.max_length)
            if res != 0:
                values = np.vstack(
                    (
                        values[..., :-res].reshape(n_segments, -1),
                        values[..., -self.max_length :],
                    )
                )
            else:
                values = values.reshape(n_segments * values.shape[0], -1)
        # if values.ndim == 1:
        #     values = values[np.newaxis, ...]

        values = self.feature_extractor(
            values,
            sampling_rate=self.config.fs,
            max_length=self.max_length,
            truncation=True,
        )
        values = [
            BatchFeature(
                dict(
                    input_values=values.input_values[0][idx],
                    # input_length=values.input_values[0].shape[-1],
                )
            )
            for idx in range(values.input_values[0].shape[0])
            if values.input_values[0].shape[-1] >= self.min_length
        ]

        return values

    def extra_repr_keys(self) -> List[str]:
        return [
            "reader",
            "ppm",
        ]

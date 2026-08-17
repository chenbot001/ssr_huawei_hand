"""
Dataset class for the SSR-Huawei UR5 + RyHand pick-and-place task.

Loads a pre-merged zarr store (produced by scripts/preprocess_dataset.py)
that already conforms to the ReplayBuffer layout:

    ssr_pickplace_dataset.zarr/
      data/
        arm_eef_pose        (N, 6)           float32   obs  (low_dim)
        hand_joint_angles   (N, 15)          float32   obs  (low_dim)
        camera_env          (N, 240, 320, 3) uint8     obs  (rgb)
        camera_wrist        (N, 240, 320, 3) uint8     obs  (rgb)
        action              (N, 21)          float32   action
      meta/
        episode_ends        (E,)             int64

No raw-data conversion step is needed — the zarr is read directly via
ReplayBuffer.copy_from_path (in-memory) or create_from_path (on-disk).
"""

from typing import Dict

import copy
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask,
)
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import (
    LinearNormalizer, SingleFieldLinearNormalizer,
)
from diffusion_policy.common.normalize_util import get_image_range_normalizer


def _get_val_mask_stratified_length(
    episode_ends: np.ndarray,
    val_ratio: float,
    n_strata: int = 4,
    seed: int = 42,
) -> np.ndarray:
    """Stratified val mask using episode length as a proxy for episode type.

    Divides episodes into ``n_strata`` buckets by length percentile and samples
    ``val_ratio`` from each bucket independently.  This ensures short episodes
    (subaction / failure-recovery demos) and long episodes (full end-to-end)
    are proportionally represented in both train and val splits even when the
    dataset is a composite of unlabelled episode types.
    """
    lengths = np.diff(np.concatenate([[0], episode_ends]))
    n_episodes = len(lengths)
    rng = np.random.default_rng(seed=seed)

    val_mask = np.zeros(n_episodes, dtype=bool)
    # Assign each episode to a stratum by length percentile
    percentile_edges = np.percentile(lengths, np.linspace(0, 100, n_strata + 1))
    # Force unique edges to handle ties at boundaries
    percentile_edges = np.unique(percentile_edges)

    for i in range(len(percentile_edges) - 1):
        lo, hi = percentile_edges[i], percentile_edges[i + 1]
        if i == len(percentile_edges) - 2:
            # last stratum is inclusive on the right
            stratum_idxs = np.where((lengths >= lo) & (lengths <= hi))[0]
        else:
            stratum_idxs = np.where((lengths >= lo) & (lengths < hi))[0]
        if len(stratum_idxs) == 0:
            continue
        n_val = max(1, round(len(stratum_idxs) * val_ratio))
        n_val = min(n_val, len(stratum_idxs) - 1) if len(stratum_idxs) > 1 else 0
        if n_val > 0:
            chosen = rng.choice(stratum_idxs, size=n_val, replace=False)
            val_mask[chosen] = True

    return val_mask


class SSRPickPlaceImageDataset(BaseImageDataset):
    def __init__(
        self,
        shape_meta: dict,
        dataset_path: str,
        horizon: int = 1,
        pad_before: int = 0,
        pad_after: int = 0,
        n_obs_steps: int = None,
        n_latency_steps: int = 0,
        seed: int = 42,
        val_ratio: float = 0.02,
        max_train_episodes: int = None,
        load_into_memory: bool = True,
        val_split_strategy: str = "stratified_length",
        val_split_n_strata: int = 4,
    ):
        super().__init__()

        obs_shape_meta = shape_meta["obs"]
        rgb_keys = []
        lowdim_keys = []
        for key, attr in obs_shape_meta.items():
            obs_type = attr.get("type", "low_dim")
            if obs_type == "rgb":
                rgb_keys.append(key)
            elif obs_type == "low_dim":
                lowdim_keys.append(key)

        all_data_keys = rgb_keys + lowdim_keys + ["action"]

        if load_into_memory:
            replay_buffer = ReplayBuffer.copy_from_path(
                dataset_path, keys=all_data_keys)
        else:
            replay_buffer = ReplayBuffer.create_from_path(
                dataset_path, mode="r")

        key_first_k: dict = {}
        if n_obs_steps is not None:
            for key in rgb_keys + lowdim_keys:
                key_first_k[key] = n_obs_steps

        if val_split_strategy == "stratified_length":
            val_mask = _get_val_mask_stratified_length(
                episode_ends=replay_buffer.episode_ends,
                val_ratio=val_ratio,
                n_strata=val_split_n_strata,
                seed=seed,
            )
        else:
            val_mask = get_val_mask(
                n_episodes=replay_buffer.n_episodes,
                val_ratio=val_ratio,
                seed=seed,
            )
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask,
            max_n=max_train_episodes,
            seed=seed,
        )

        sampler = SequenceSampler(
            replay_buffer=replay_buffer,
            sequence_length=horizon + n_latency_steps,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k=key_first_k,
        )

        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.n_obs_steps = n_obs_steps
        self.val_mask = val_mask
        self.horizon = horizon
        self.n_latency_steps = n_latency_steps
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon + self.n_latency_steps,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=self.val_mask,
        )
        val_set.val_mask = ~self.val_mask
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        normalizer["action"] = SingleFieldLinearNormalizer.create_fit(
            self.replay_buffer["action"])

        for key in self.lowdim_keys:
            normalizer[key] = SingleFieldLinearNormalizer.create_fit(
                self.replay_buffer[key])

        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()

        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer["action"][:])

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)
        data = self.sampler.sample_sequence(idx)

        T_slice = slice(self.n_obs_steps)

        obs_dict: dict = {}
        for key in self.rgb_keys:
            # (T, H, W, C) uint8 -> (T, C, H, W) float32 [0, 1]
            obs_dict[key] = np.moveaxis(
                data[key][T_slice], -1, 1).astype(np.float32) / 255.0
            del data[key]

        for key in self.lowdim_keys:
            obs_dict[key] = data[key][T_slice].astype(np.float32)
            del data[key]

        action = data["action"].astype(np.float32)
        if self.n_latency_steps > 0:
            action = action[self.n_latency_steps:]

        torch_data = {
            "obs": dict_apply(obs_dict, torch.from_numpy),
            "action": torch.from_numpy(action),
        }
        return torch_data

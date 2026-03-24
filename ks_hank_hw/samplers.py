from __future__ import annotations

import numpy as np

from .config import ModelConfig, SamplingConfig


class BaseSampler:
    def __init__(self, model_cfg: ModelConfig, sampling_cfg: SamplingConfig):
        self.mc = model_cfg
        self.sc = sampling_cfg
        p_low = self.mc.lam1 / (self.mc.lam1 + self.mc.lam2)
        self.plam = [p_low, 1.0 - p_low]

    @staticmethod
    def _lhs(dim: int, n_points: int):
        if dim <= 0:
            return np.zeros((n_points, 0), dtype=np.float64)
        if n_points <= 0:
            return np.zeros((0, dim), dtype=np.float64)
        cut = np.linspace(0.0, 1.0, n_points + 1)
        u = np.random.rand(n_points, dim)
        a = cut[:-1][:, None]
        b = cut[1:][:, None]
        samples = a + (b - a) * u
        lhs = np.empty_like(samples)
        for j in range(dim):
            lhs[:, j] = samples[np.random.permutation(n_points), j]
        return lhs

    def _sample_y(self, n_points: int):
        return np.random.choice(np.array([0.0, 1.0]), size=(n_points, self.mc.n_pop), p=self.plam)

    def _sample_z(self, n_points: int):
        return (self.mc.z_max - self.mc.z_min) * np.random.rand(n_points, 1) + self.mc.z_min

    def _sample_pi(self, n_points: int):
        return (self.mc.pi_max - self.mc.pi_min) * np.random.rand(n_points, 1) + self.mc.pi_min

    def _per_capita_asset_supply(self) -> float:
        # Zero-net-bond benchmark in the current model.
        return 0.0

    @staticmethod
    def _project_row_to_bounded_sum(
        raw_row: np.ndarray,
        target_sum: float,
        lower: float,
        upper: float,
        *,
        max_iter: int = 80,
        tol: float = 1e-10,
    ) -> np.ndarray:
        n = raw_row.size
        min_sum = n * lower
        max_sum = n * upper
        if target_sum < min_sum - tol or target_sum > max_sum + tol:
            raise ValueError(
                f"Infeasible bounded-sum target {target_sum:.6f} for n={n}, bounds=({lower}, {upper})."
            )

        # Find lambda such that sum(clip(raw - lambda, lower, upper)) == target_sum.
        lam_lo = float(np.min(raw_row - upper))
        lam_hi = float(np.max(raw_row - lower))

        def f(lam: float) -> float:
            return float(np.sum(np.clip(raw_row - lam, lower, upper)) - target_sum)

        f_lo = f(lam_lo)
        f_hi = f(lam_hi)
        if f_lo < 0.0:
            lam_lo = lam_lo - abs(f_lo) - 1.0
            f_lo = f(lam_lo)
        if f_hi > 0.0:
            lam_hi = lam_hi + abs(f_hi) + 1.0
            f_hi = f(lam_hi)
        if f_lo < 0.0 or f_hi > 0.0:
            raise RuntimeError("Failed to bracket bounded-sum projection root.")

        for _ in range(max_iter):
            lam_mid = 0.5 * (lam_lo + lam_hi)
            f_mid = f(lam_mid)
            if abs(f_mid) <= tol:
                return np.clip(raw_row - lam_mid, lower, upper)
            if f_mid > 0.0:
                lam_lo = lam_mid
            else:
                lam_hi = lam_mid
        lam_mid = 0.5 * (lam_lo + lam_hi)
        return np.clip(raw_row - lam_mid, lower, upper)

    def _project_rows_to_bounded_sum(self, raw: np.ndarray, target_sums: np.ndarray):
        projected = np.empty_like(raw, dtype=np.float64)
        lo = float(self.mc.a_min)
        hi = float(self.mc.a_max)
        for i in range(raw.shape[0]):
            projected[i, :] = self._project_row_to_bounded_sum(raw[i, :], float(target_sums[i]), lo, hi)
        return projected

    def _sample_other_assets(self, n_points: int, target_other_sum: np.ndarray | None = None):
        n_other = self.mc.n_pop - 1
        if n_other <= 0:
            if target_other_sum is not None:
                target_other_sum = np.asarray(target_other_sum, dtype=np.float64).reshape(-1)
                if np.any(np.abs(target_other_sum) > 1e-8):
                    raise ValueError("Cannot satisfy non-zero asset-clearing target with n_pop <= 1.")
            return np.zeros((n_points, 0), dtype=np.float32)
        raw = self.mc.a_min + (self.mc.a_max - self.mc.a_min) * self._lhs(n_other, n_points)
        if target_other_sum is None:
            # Backward-compatible fallback (not exact clearing).
            raw = raw - np.mean(raw, axis=1, keepdims=True)
            raw = np.clip(raw, self.mc.a_min, self.mc.a_max)
            return raw.astype(np.float32)
        target = np.asarray(target_other_sum, dtype=np.float64).reshape(-1)
        if target.size != n_points:
            raise ValueError(f"target_other_sum size {target.size} != n_points {n_points}")
        a = self._project_rows_to_bounded_sum(raw, target)
        return a.astype(np.float32)

    def _sample_stationary_other_assets(
        self, n_points: int, a_vals: np.ndarray, g: np.ndarray, target_other_sum: np.ndarray | None = None
    ):
        n_other = self.mc.n_pop - 1
        if n_other <= 0:
            if target_other_sum is not None:
                target_other_sum = np.asarray(target_other_sum, dtype=np.float64).reshape(-1)
                if np.any(np.abs(target_other_sum) > 1e-8):
                    raise ValueError("Cannot satisfy non-zero asset-clearing target with n_pop <= 1.")
            return np.zeros((n_points, 0), dtype=np.float32)

        y_other = np.random.choice(np.arange(0, 2), size=(n_points, n_other), p=self.plam)
        g_low = np.asarray(g[0], dtype=np.float64)
        g_high = np.asarray(g[1], dtype=np.float64)
        g_low = g_low / np.sum(g_low)
        g_high = g_high / np.sum(g_high)
        draws_low = np.random.choice(a_vals, size=(n_points, n_other), p=g_low)
        draws_high = np.random.choice(a_vals, size=(n_points, n_other), p=g_high)
        raw = np.where(y_other == 0, draws_low, draws_high).astype(np.float64)
        if target_other_sum is None:
            # Backward-compatible fallback (not exact clearing).
            raw = raw - np.mean(raw, axis=1, keepdims=True)
            raw = np.clip(raw, self.mc.a_min, self.mc.a_max)
            return raw.astype(np.float32)
        target = np.asarray(target_other_sum, dtype=np.float64).reshape(-1)
        if target.size != n_points:
            raise ValueError(f"target_other_sum size {target.size} != n_points {n_points}")
        a = self._project_rows_to_bounded_sum(raw, target)
        return a.astype(np.float32)

    def sample(self, n_points: int):
        raise NotImplementedError

    def training_sample_size(self, default_n_points: int) -> int:
        return int(default_n_points)

    def batch_sample_size(self, default_n_points: int) -> int:
        return int(default_n_points)

    def sample_main(self, n_points: int):
        return self.sample(n_points)

    def sample_main_with_interval_ids(self, n_points: int):
        x, agg = self.sample_main(n_points)
        return x, agg, None

    def sample_batch_indices(self, interval_ids, n_points: int | None = None):
        sample_size = int(n_points if n_points is not None else self.batch_sample_size(0))
        if interval_ids is None or sample_size <= 0:
            return np.array([], dtype=np.int64)
        total_points = int(len(interval_ids))
        if total_points <= 0:
            return np.array([], dtype=np.int64)
        return np.random.choice(np.arange(total_points), size=sample_size, replace=True).astype(np.int64)

    def sample_focus_region(self, n_points: int, a_lo: float, a_hi: float):
        if n_points <= 0:
            n_dim = 2 * self.mc.n_pop
            return np.zeros((0, n_dim), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)
        z = self._sample_z(n_points).astype(np.float32)
        pi = self._sample_pi(n_points).astype(np.float32)
        if np.isclose(a_lo, a_hi):
            a_i = np.full((n_points, 1), a_lo, dtype=np.float32)
        else:
            a_i = (a_lo + (a_hi - a_lo) * self._lhs(1, n_points)).astype(np.float32)
        target_total_assets = float(self.mc.n_pop) * float(self._per_capita_asset_supply())
        target_other_sum = target_total_assets - a_i[:, 0].astype(np.float64)
        a_other = self._sample_other_assets(n_points, target_other_sum=target_other_sum)
        a = np.hstack([a_i, a_other])
        y = self._sample_y(n_points)
        x = np.hstack([a, y]).astype(np.float32)
        agg = np.hstack([z, pi]).astype(np.float32)
        return x, agg

    def sample_active_learning(self):
        upper = max(self.mc.a_min, self.sc.al_region_upper_share * self.mc.a_max)
        return self.sample_focus_region(int(self.sc.al_points), self.mc.a_min, upper)

    def sample_boundary(self):
        return self.sample_focus_region(int(self.sc.bc_points), self.mc.a_min, self.mc.a_min)

    def pretrain_sample(self, n_points: int, a_vals: np.ndarray, g: np.ndarray):
        # Keep a_vals/g in the signature for runner compatibility,
        # but pretraining now uses the same synthetic exact-clearing sampler.
        _ = (a_vals, g)
        z = np.full((n_points, 1), self.mc.z_bar, dtype=np.float32)
        pi = np.zeros((n_points, 1), dtype=np.float32)
        a_i = (self.mc.a_min + (self.mc.a_max - self.mc.a_min) * self._lhs(1, n_points)).astype(np.float32)
        target_total_assets = float(self.mc.n_pop) * float(self._per_capita_asset_supply())
        target_other_sum = target_total_assets - a_i[:, 0].astype(np.float64)
        a_other = self._sample_other_assets(n_points, target_other_sum=target_other_sum)
        a = np.hstack([a_i, a_other])
        y = self._sample_y(n_points)
        x = np.hstack([a, y]).astype(np.float32)
        agg = np.hstack([z, pi]).astype(np.float32)
        return x, agg

    @staticmethod
    def sample_aux_batch_indices(total_points: int, batch_points: int):
        total = int(total_points)
        batch = int(batch_points)
        if total <= 0 or batch <= 0:
            return np.array([], dtype=np.int64)
        return np.random.choice(np.arange(total), size=batch, replace=True).astype(np.int64)


class UniformSampler(BaseSampler):
    """Uniform interval sampler in own assets."""

    def __init__(self, model_cfg: ModelConfig, sampling_cfg: SamplingConfig):
        super().__init__(model_cfg, sampling_cfg)
        self.edges = np.linspace(model_cfg.a_min, model_cfg.a_max, sampling_cfg.num_intervals + 1)
        self.interval_point_counts = np.ones(sampling_cfg.num_intervals, dtype=np.int64) * int(
            sampling_cfg.points_per_interval
        )
        self.interval_batch_counts = np.ones(sampling_cfg.num_intervals, dtype=np.int64) * int(
            sampling_cfg.batch_points_per_interval
        )

    def training_sample_size(self, default_n_points: int) -> int:
        return int(np.sum(self.interval_point_counts))

    def batch_sample_size(self, default_n_points: int) -> int:
        return int(np.sum(self.interval_batch_counts))

    def sample_main_with_interval_ids(self, n_points: int):
        counts = self.interval_point_counts.copy()
        total = int(np.sum(counts))
        if int(n_points) != total:
            counts = np.ones(self.sc.num_intervals, dtype=np.int64) * max(int(n_points) // self.sc.num_intervals, 0)
            remainder = int(n_points) - int(np.sum(counts))
            if remainder > 0:
                counts[:remainder] += 1
        total = int(np.sum(counts))
        z = self._sample_z(total).astype(np.float32)
        pi = self._sample_pi(total).astype(np.float32)
        a_i_blocks = []
        interval_ids = []
        for j, n_j in enumerate(counts):
            if n_j <= 0:
                continue
            lo, hi = self.edges[j], self.edges[j + 1]
            a_i_blocks.append((lo + (hi - lo) * self._lhs(1, int(n_j))).astype(np.float32))
            interval_ids.append(np.full(int(n_j), j, dtype=np.int64))
        a_i = np.vstack(a_i_blocks) if a_i_blocks else np.zeros((0, 1), dtype=np.float32)
        target_total_assets = float(self.mc.n_pop) * float(self._per_capita_asset_supply())
        target_other_sum = target_total_assets - a_i[:, 0].astype(np.float64)
        a_other = self._sample_other_assets(total, target_other_sum=target_other_sum)
        ids = np.concatenate(interval_ids) if interval_ids else np.zeros(0, dtype=np.int64)
        a = np.hstack([a_i, a_other])
        y = self._sample_y(total)
        x = np.hstack([a, y]).astype(np.float32)
        agg = np.hstack([z, pi]).astype(np.float32)
        return x, agg, ids

    def sample_main(self, n_points: int):
        x, agg, _ = self.sample_main_with_interval_ids(n_points)
        return x, agg

    def sample(self, n_points: int):
        return self.sample_main(n_points)

    def sample_batch_indices(self, interval_ids, n_points: int | None = None):
        if interval_ids is None:
            return super().sample_batch_indices(interval_ids, n_points)
        interval_ids = np.asarray(interval_ids)
        selected = []
        for j, n_j in enumerate(self.interval_batch_counts):
            if n_j <= 0:
                continue
            candidates = np.flatnonzero(interval_ids == j)
            if candidates.size == 0:
                continue
            selected.append(np.random.choice(candidates, size=int(n_j), replace=True))
        if not selected:
            return np.array([], dtype=np.int64)
        return np.concatenate(selected).astype(np.int64)


class AdaptiveIntervalSampler(UniformSampler):
    """Adaptive interval sampler preserving PS4 logic."""

    def __init__(self, model_cfg: ModelConfig, sampling_cfg: SamplingConfig):
        super().__init__(model_cfg, sampling_cfg)
        self._point_budget = int(np.sum(self.interval_point_counts))
        self._batch_budget = int(np.sum(self.interval_batch_counts))
        self.interval_priority = self.interval_point_counts.astype(np.float64).copy()
        self._sync_distribution()

    @staticmethod
    def _allocate_budget(total: int, weights: np.ndarray):
        total = int(total)
        if total <= 0:
            return np.zeros_like(weights, dtype=np.int64)
        scaled = np.asarray(weights, dtype=np.float64) * total
        counts = np.floor(scaled).astype(np.int64)
        remainder = total - int(np.sum(counts))
        if remainder > 0:
            order = np.argsort(-(scaled - counts))
            counts[order[:remainder]] += 1
        return counts

    def _sync_distribution(self):
        total = float(np.sum(self.interval_priority))
        if total <= 0:
            self.interval_priority = np.ones(self.sc.num_intervals, dtype=np.float64)
            total = float(np.sum(self.interval_priority))
        self.interval_weights = self.interval_priority / total
        self.interval_point_counts = self._allocate_budget(self._point_budget, self.interval_weights)
        self.interval_batch_counts = self._allocate_budget(self._batch_budget, self.interval_weights)

    def add_points(self, center_interval: int, points_to_add: int):
        span = max(int(self.sc.adaptive_neighbor_span), 0)
        decay = float(self.sc.adaptive_neighbor_decay)
        for offset in range(-span, span + 1):
            j = center_interval + offset
            if 0 <= j < self.sc.num_intervals:
                increment = int(points_to_add * (decay ** abs(offset)))
                if increment > 0:
                    self.interval_priority[j] += increment
        self._sync_distribution()

    def sample_main_with_interval_ids(self, n_points: int):
        counts = self.interval_point_counts.copy()
        total = int(np.sum(counts))
        if int(n_points) != total:
            counts = np.random.multinomial(int(n_points), self.interval_weights)
            total = int(np.sum(counts))
        z = self._sample_z(total).astype(np.float32)
        pi = self._sample_pi(total).astype(np.float32)
        a_i_blocks = []
        interval_ids = []
        for j, n_j in enumerate(counts):
            if n_j <= 0:
                continue
            lo, hi = self.edges[j], self.edges[j + 1]
            a_i_blocks.append((lo + (hi - lo) * self._lhs(1, int(n_j))).astype(np.float32))
            interval_ids.append(np.full(int(n_j), j, dtype=np.int64))
        a_i = np.vstack(a_i_blocks) if a_i_blocks else np.zeros((0, 1), dtype=np.float32)
        target_total_assets = float(self.mc.n_pop) * float(self._per_capita_asset_supply())
        target_other_sum = target_total_assets - a_i[:, 0].astype(np.float64)
        a_other = self._sample_other_assets(total, target_other_sum=target_other_sum)
        ids = np.concatenate(interval_ids) if interval_ids else np.zeros(0, dtype=np.int64)
        a = np.hstack([a_i, a_other])
        y = self._sample_y(total)
        x = np.hstack([a, y]).astype(np.float32)
        agg = np.hstack([z, pi]).astype(np.float32)
        return x, agg, ids

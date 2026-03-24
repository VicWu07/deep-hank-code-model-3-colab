from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .config import RunConfig
from .losses import weighted_loss


class Trainer:
    def __init__(self, cfg: RunConfig, model, sampler, residual_op, positive_bound: float | None = None):
        self.cfg = cfg
        self.model = model
        self.sampler = sampler
        self.residual_op = residual_op
        self.positive_bound = positive_bound
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device).float()

    def _to_tensor(self, x_np, agg_np):
        x = torch.tensor(x_np, dtype=torch.float32, device=self.device)
        agg = torch.tensor(agg_np, dtype=torch.float32, device=self.device)
        return x, agg

    def _default_train_sample_size(self) -> int:
        return int(self.cfg.sampling.num_intervals * self.cfg.sampling.points_per_interval)

    def _train_sample_size(self) -> int:
        sample_size_fn = getattr(self.sampler, "training_sample_size", None)
        if callable(sample_size_fn):
            return int(sample_size_fn(self._default_train_sample_size()))
        return self._default_train_sample_size()

    def _default_main_batch_size(self) -> int:
        return int(self.cfg.sampling.num_intervals * self.cfg.sampling.batch_points_per_interval)

    def _main_batch_size(self) -> int:
        batch_size_fn = getattr(self.sampler, "batch_sample_size", None)
        if callable(batch_size_fn):
            return int(batch_size_fn(self._default_main_batch_size()))
        return self._default_main_batch_size()

    def _interval_scores(self):
        sample_with_ids = getattr(self.sampler, "sample_main_with_interval_ids", None)
        if not callable(sample_with_ids):
            return None
        n_points = self._train_sample_size()
        x_np, agg_np, interval_ids = sample_with_ids(n_points)
        if interval_ids.size == 0:
            return None
        x, agg = self._to_tensor(x_np, agg_np)
        agg.requires_grad_(True)
        residuals = self.residual_op(self.model, x, agg).detach().cpu().numpy().reshape(-1)
        outputs = self.model(x, agg).detach().cpu().numpy().reshape(-1)
        abs_sq = np.square(residuals)
        denom = np.maximum(np.abs(outputs), float(self.cfg.sampling.adaptive_relative_eps))
        relative_sq = np.square(residuals / denom)
        rel_weight = max(float(getattr(self.cfg.sampling, "adaptive_relative_score_weight", 0.0)), 0.0)
        scores = np.full(int(self.cfg.sampling.num_intervals), -np.inf, dtype=np.float64)
        for j in range(scores.size):
            mask = interval_ids == j
            if np.any(mask):
                # Primary signal: interval residual MSE (aligned with training objective).
                # Small relative-error term acts as tie-break when MSE levels are close.
                scores[j] = float(np.mean(abs_sq[mask]) + rel_weight * np.mean(relative_sq[mask]))
        return scores

    def _select_adaptive_interval(self, scores: np.ndarray):
        if scores.size == 0:
            return None
        lower = max(int(self.cfg.sampling.adaptive_skip_lower_intervals), 0)
        upper = max(int(self.cfg.sampling.adaptive_skip_upper_intervals), 0)
        lo = min(lower, scores.size)
        hi = max(lo, scores.size - upper)
        candidate = scores[lo:hi] if hi > lo else scores
        offset = lo if hi > lo else 0
        filtered = candidate.copy()
        filtered[~np.isfinite(filtered)] = -np.inf
        if filtered.size == 0 or np.all(filtered == -np.inf):
            return None
        return int(offset + np.argmax(filtered))

    def _maybe_update_adaptive_sampler(self, epoch: int, total_epochs: int):
        if not hasattr(self.sampler, "add_points"):
            return None
        if int(self.cfg.sampling.add_points) <= 0:
            return None
        update_every = max(int(self.cfg.sampling.adaptive_update_every_epochs), 1)
        burn_in_share = getattr(self.cfg.sampling, "adaptive_burn_in_share", None)
        if burn_in_share is None:
            burn_in = max(int(self.cfg.sampling.adaptive_burn_in_epochs), 0)
        else:
            burn_in = max(int(round(max(int(total_epochs), 1) * float(burn_in_share))), 0)
        if epoch <= burn_in or epoch % update_every != 0:
            return None
        scores = self._interval_scores()
        if scores is None:
            return None
        interval = self._select_adaptive_interval(scores)
        if interval is None:
            return None
        self.sampler.add_points(interval, int(self.cfg.sampling.add_points))
        total_points = getattr(self.sampler, "training_sample_size", None)
        return {
            "adaptive_interval": interval,
            "adaptive_interval_score": float(scores[interval]),
            "adaptive_total_points": int(total_points(self._default_train_sample_size()))
            if callable(total_points)
            else self._train_sample_size(),
        }

    def _build_epoch_samples(self):
        main_with_ids = getattr(self.sampler, "sample_main_with_interval_ids", None)
        if callable(main_with_ids):
            x_main_np, agg_main_np, interval_ids = main_with_ids(self._train_sample_size())
        else:
            x_main_np, agg_main_np = self.sampler.sample(self._train_sample_size())
            interval_ids = None

        x_al_np, agg_al_np = self.sampler.sample_active_learning()
        x_bc_np, agg_bc_np = self.sampler.sample_boundary()
        return {
            "main": (x_main_np, agg_main_np, interval_ids),
            "al": (x_al_np, agg_al_np),
            "bc": (x_bc_np, agg_bc_np),
        }

    def _sample_epoch_batch(self, epoch_samples):
        x_main_np, agg_main_np, interval_ids = epoch_samples["main"]
        x_al_np, agg_al_np = epoch_samples["al"]
        x_bc_np, agg_bc_np = epoch_samples["bc"]

        main_idx = self.sampler.sample_batch_indices(interval_ids, self._main_batch_size())
        al_idx = self.sampler.sample_aux_batch_indices(len(x_al_np), int(self.cfg.sampling.al_batch_points))
        bc_idx = self.sampler.sample_aux_batch_indices(len(x_bc_np), int(self.cfg.sampling.bc_batch_points))

        x_blocks = []
        agg_blocks = []

        if main_idx.size > 0:
            x_blocks.append(x_main_np[main_idx, :])
            agg_blocks.append(agg_main_np[main_idx, :])
        if al_idx.size > 0:
            x_blocks.append(x_al_np[al_idx, :])
            agg_blocks.append(agg_al_np[al_idx, :])
        if bc_idx.size > 0:
            x_blocks.append(x_bc_np[bc_idx, :])
            agg_blocks.append(agg_bc_np[bc_idx, :])

        if not x_blocks:
            n_dim = 2 * self.cfg.model.n_pop
            return np.zeros((0, n_dim), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)
        return np.vstack(x_blocks).astype(np.float32), np.vstack(agg_blocks).astype(np.float32)

    def pretrain(self, epochs: int, target_fn, sample_fn=None, return_history: bool = False):
        opt = torch.optim.Adam(self.model.parameters(), lr=self.cfg.train.lr_pretrain)
        tic = perf_counter()
        history = []
        for epoch in range(1, epochs + 1):
            if sample_fn is not None:
                x_np, agg_np = sample_fn(100)
            else:
                x_np, agg_np = self.sampler.sample(512)
            x, agg = self._to_tensor(x_np, agg_np)
            pred = self.model(x, agg)
            tgt = target_fn(x, agg)
            loss = torch.mean(torch.square(pred - tgt))
            opt.zero_grad()
            loss.backward()
            opt.step()
            if return_history:
                history.append({"epoch": epoch, "loss": float(loss.detach().cpu())})
        elapsed = perf_counter() - tic
        if return_history:
            return elapsed, history
        return elapsed

    @staticmethod
    def _write_monitor(history: list[dict], monitor_path: Path | None):
        if monitor_path is None:
            return

        tail = history[-5:]
        lines = [
            "latest_five_epochs",
            "epoch,loss,residual,shape,dvdz,dvda,epoch_time_sec,elapsed_sec",
        ]
        for item in tail:
            lines.append(
                "{epoch},{loss:.6e},{residual:.6e},{shape:.6e},{dvdz:.6e},{dvda:.6e},{epoch_time_sec:.4f},{elapsed_sec:.2f}".format(
                    **item
                )
            )
        monitor_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    @staticmethod
    def _save_checkpoint(
        checkpoint_path: Path,
        cfg: RunConfig,
        model,
        optimizer,
        metrics: dict,
    ):
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": metrics["epoch"],
                "config": asdict(cfg),
                "metrics": metrics,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            checkpoint_path,
        )

    def train(
        self,
        epochs: int,
        monitor_path: str | Path | None = None,
        checkpoint_dir: str | Path | None = None,
    ):
        monitor_file = Path(monitor_path) if monitor_path is not None else None
        checkpoint_root = Path(checkpoint_dir) if checkpoint_dir is not None else None
        opt = torch.optim.Adam(self.model.parameters(), lr=self.cfg.train.lr_train)
        scheduler = None
        if str(getattr(self.cfg.train, "lr_scheduler", "none")).lower() == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt,
                mode="min",
                factor=float(self.cfg.train.lr_plateau_factor),
                patience=int(self.cfg.train.lr_plateau_patience),
                threshold=float(self.cfg.train.lr_plateau_threshold),
                cooldown=int(self.cfg.train.lr_plateau_cooldown),
                min_lr=float(self.cfg.train.lr_plateau_min),
            )
        history = []
        tic = perf_counter()
        run_start = tic
        stop_reason = "max_epochs"
        stop_epoch = epochs
        best_residual = float("inf")
        for epoch in range(1, epochs + 1):
            epoch_start = perf_counter()
            epoch_samples = self._build_epoch_samples()
            batch_losses = []
            batch_residuals = []
            batch_shapes = []
            batch_dvdz = []
            batch_dvda = []
            last_batch_size = 0

            for _ in range(max(int(self.cfg.train.n_batch), 1)):
                x_np, agg_np = self._sample_epoch_batch(epoch_samples)
                last_batch_size = int(x_np.shape[0])
                if last_batch_size == 0:
                    continue
                x, agg = self._to_tensor(x_np, agg_np)
                x.requires_grad_(True)
                agg.requires_grad_(True)
                residuals = self.residual_op(self.model, x, agg)
                outputs = self.model(x, agg)
                dV_dagg = torch.autograd.grad(
                    outputs,
                    agg,
                    grad_outputs=torch.ones_like(outputs),
                    create_graph=True,
                )[0]
                dV_dx = torch.autograd.grad(
                    outputs,
                    x,
                    grad_outputs=torch.ones_like(outputs),
                    create_graph=True,
                )[0]
                dvdz = dV_dagg[:, :1]
                dvda = dV_dx[:, :1]
                loss, l_res, l_shape, l_dvdz, l_dvda = weighted_loss(
                    residuals=residuals,
                    outputs=outputs,
                    dvdz=dvdz,
                    dvda=dvda,
                    w_residual=self.cfg.train.weight_residual,
                    w_shape=self.cfg.train.weight_shape,
                    w_dvdz=self.cfg.train.weight_dvdz,
                    w_dvda=self.cfg.train.weight_dvda,
                    positive_bound=self.positive_bound,
                )
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.clip_grad)
                opt.step()

                batch_losses.append(float(loss.detach().cpu()))
                batch_residuals.append(float(l_res.detach().cpu()))
                batch_shapes.append(float(l_shape.detach().cpu()))
                batch_dvdz.append(float(l_dvdz.detach().cpu()))
                batch_dvda.append(float(l_dvda.detach().cpu()))

            if not batch_losses:
                history.append(
                    {
                        "epoch": epoch,
                        "loss": float("nan"),
                        "residual": float("nan"),
                        "shape": float("nan"),
                        "dvdz": float("nan"),
                        "dvda": float("nan"),
                        "sample_points": 0,
                        "epoch_time_sec": perf_counter() - epoch_start,
                        "elapsed_sec": perf_counter() - run_start,
                    }
                )
                continue
            history.append(
                {
                    "epoch": epoch,
                    "loss": float(np.mean(batch_losses)),
                    "residual": float(np.mean(batch_residuals)),
                    "shape": float(np.mean(batch_shapes)),
                    "dvdz": float(np.mean(batch_dvdz)),
                    "dvda": float(np.mean(batch_dvda)),
                    "main_sample_points": int(epoch_samples["main"][0].shape[0]),
                    "al_sample_points": int(epoch_samples["al"][0].shape[0]),
                    "bc_sample_points": int(epoch_samples["bc"][0].shape[0]),
                    "batch_points": int(last_batch_size),
                    "sample_points": int(
                        epoch_samples["main"][0].shape[0]
                        + epoch_samples["al"][0].shape[0]
                        + epoch_samples["bc"][0].shape[0]
                    ),
                    "epoch_time_sec": perf_counter() - epoch_start,
                    "elapsed_sec": perf_counter() - run_start,
                }
            )
            current = history[-1]
            if scheduler is not None:
                metric_name = str(self.cfg.train.lr_plateau_metric).lower()
                metric_value = current["residual"] if metric_name == "residual" else current["loss"]
                scheduler.step(metric_value)
                current["lr"] = float(opt.param_groups[0]["lr"])
            if checkpoint_root is not None and epoch % 10 == 0 and current["residual"] < best_residual:
                best_residual = current["residual"]
                self._save_checkpoint(
                    checkpoint_root / "best_residual.pt",
                    self.cfg,
                    self.model,
                    opt,
                    current,
                )
            if epoch % 10 == 0:
                self._write_monitor(history, monitor_file)
                toc = perf_counter()
                print(f"[{epoch}] loss={history[-1]['loss']:.4e}  dt={toc - tic:.2f}s")
                tic = toc
            if checkpoint_root is not None and epoch % int(self.cfg.train.checkpoint_every_epochs) == 0:
                self._save_checkpoint(
                    checkpoint_root / f"epoch_{epoch:06d}.pt",
                    self.cfg,
                    self.model,
                    opt,
                    current,
                )
            adaptive_update = self._maybe_update_adaptive_sampler(epoch, epochs)
            if adaptive_update is not None:
                current.update(adaptive_update)
            threshold = self.cfg.train.stop_residual_threshold
            window = max(int(self.cfg.train.stop_window), 1)
            if (
                threshold is not None
                and epoch >= max(int(self.cfg.train.min_train_epochs), window)
            ):
                tail = history[-window:]
                rolling_residual = float(np.mean([item["residual"] for item in tail]))
                if rolling_residual <= threshold:
                    stop_reason = "residual_threshold"
                    stop_epoch = epoch
                    print(
                        f"Early stop at epoch {epoch}: "
                        f"mean residual over last {window} epochs = {rolling_residual:.4e}"
                    )
                    break
        if history:
            history[-1]["stop_reason"] = stop_reason
            history[-1]["stop_epoch"] = stop_epoch
            self._write_monitor(history, monitor_file)
            if checkpoint_root is not None:
                self._save_checkpoint(
                    checkpoint_root / "final.pt",
                    self.cfg,
                    self.model,
                    opt,
                    history[-1],
                )
        return history

    @staticmethod
    def dump_config(cfg: RunConfig):
        return asdict(cfg)

    @staticmethod
    def fit_intervals_for_adaptive_sampler(history: list[dict], k: int):
        if not history:
            return np.ones(k) / k
        scores = np.ones(k, dtype=np.float64)
        for item in reversed(history):
            interval = item.get("adaptive_interval")
            score = item.get("adaptive_interval_score")
            if interval is not None and score is not None:
                scores[int(interval)] = float(score)
        scores = np.maximum(scores, 1e-12)
        return scores / scores.sum()

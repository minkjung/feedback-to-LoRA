"""Training loop: KL(student || teacher) over original + related queries."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import FeedbackDataset, collate_singletons
from .hypernetwork import FeedbackToLoRA
from .losses import kl_distillation_loss
from .target_model import TargetModel


@dataclass
class TrainConfig:
    learning_rate: float
    batch_size: int
    gradient_accumulation_steps: int
    max_epochs: int
    early_stopping_patience: int
    warmup_ratio: float
    max_grad_norm: float
    eval_every_steps: int
    checkpoint_dir: str
    save_every_steps: int = 200


class Trainer:
    def __init__(
        self,
        hypernetwork: FeedbackToLoRA,
        target: TargetModel,
        train_ds: FeedbackDataset,
        val_ds: FeedbackDataset,
        config: TrainConfig,
        device: str = "cuda",
    ):
        self.hypernetwork = hypernetwork.to(device)
        self.target = target.to(device)
        self.config = config
        self.device = device

        self.train_loader = DataLoader(
            train_ds,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate_singletons,
        )
        self.val_loader = DataLoader(
            val_ds,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collate_singletons,
        )

        trainable = [p for p in self.hypernetwork.parameters() if p.requires_grad]
        self.optimizer = AdamW(trainable, lr=config.learning_rate)

        total_steps = max(1, len(self.train_loader) * config.max_epochs)
        self.warmup_steps = int(config.warmup_ratio * total_steps)
        self.total_steps = total_steps

        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        self.best_val = math.inf
        self.patience = 0
        self.global_step = 0
        self.start_epoch = 0
        self._resume_wandb_id: str | None = None
        self._maybe_resume()

        self.wandb = None
        try:
            import wandb
            import os
            if os.environ.get("WANDB_API_KEY"):
                run_id = self._resume_wandb_id
                wandb.init(
                    project="feedback-to-lora",
                    config=vars(config),
                    settings=wandb.Settings(init_timeout=300),
                    id=run_id,
                    resume="allow" if run_id else None,
                )
                self.wandb = wandb
        except Exception as e:
            print(f"[wandb] disabled: {e}")

    # ---------- resume ----------

    def _maybe_resume(self) -> None:
        resume_path = Path(self.config.checkpoint_dir) / "latest.pt"
        if not resume_path.exists():
            # Local-only: latest.pt is not uploaded to HF anymore.
            return
        ckpt = torch.load(resume_path, map_location=self.device)
        # strict=False because backbone.* keys are intentionally absent (frozen)
        self.hypernetwork.load_state_dict(ckpt["model"], strict=False)
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        self.global_step = ckpt.get("step", 0)
        self.start_epoch = ckpt.get("epoch", 0)
        self.best_val = ckpt.get("best_val", math.inf)
        self.patience = ckpt.get("patience", 0)
        self._resume_wandb_id = ckpt.get("wandb_id")
        print(f"[resume] step={self.global_step}, epoch={self.start_epoch}, best_val={self.best_val:.4f}")

    def _try_download_resume(self, dest: Path) -> None:
        try:
            from huggingface_hub import hf_hub_download
            repo_id = "james-kernel/feedback-to-lora-checkpoints"
            # Hub has light (no optimizer) version
            path = hf_hub_download(repo_id=repo_id, filename="latest_light.pt", repo_type="model")
            import shutil
            shutil.copy(path, dest)
            print(f"[resume] downloaded latest.pt from hub")
        except Exception:
            pass

    # ---------- LR schedule (linear warmup + cosine decay) ----------

    def _lr_scale(self, step: int) -> float:
        if step < self.warmup_steps:
            return step / max(1, self.warmup_steps)
        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    def _set_lr(self) -> None:
        scale = self._lr_scale(self.global_step)
        for g in self.optimizer.param_groups:
            g["lr"] = self.config.learning_rate * scale

    # ---------- per-sample loss ----------

    def _sample_loss_and_backward(self, sample: dict, scale: float) -> float:
        feedback_ids = sample["feedback_ids"].unsqueeze(0).to(self.device)
        feedback_mask = sample["feedback_mask"].unsqueeze(0).to(self.device)

        lora = self.hypernetwork(feedback_ids, feedback_mask)

        queries = [sample["query"]] + list(sample["related_queries"])
        total = 0.0
        for i, q in enumerate(queries):
            s_logits = self.target.student_forward(q, lora)
            t_logits = self.target.teacher_forward(q, sample["feedback"])
            kl = kl_distillation_loss(s_logits, t_logits)
            (kl * scale / len(queries)).backward(retain_graph=(i < len(queries) - 1))
            # Report raw per-query KL (unscaled) so train/loss matches val/loss scale
            total += kl.item() / len(queries)
        return total

    # ---------- step ----------

    def train_step(self, batch: dict) -> float:
        self.hypernetwork.train()
        self.optimizer.zero_grad()
        accum = self.config.gradient_accumulation_steps
        total = 0.0
        for sample in batch["samples"]:
            total += self._sample_loss_and_backward(sample, scale=1.0 / accum)
        torch.nn.utils.clip_grad_norm_(self.hypernetwork.parameters(), self.config.max_grad_norm)
        self._set_lr()
        self.optimizer.step()
        self.global_step += 1
        return total

    @torch.no_grad()
    def validate(self) -> float:
        self.hypernetwork.eval()
        losses: list[float] = []
        for batch in self.val_loader:
            for sample in batch["samples"]:
                feedback_ids = sample["feedback_ids"].unsqueeze(0).to(self.device)
                feedback_mask = sample["feedback_mask"].unsqueeze(0).to(self.device)
                lora = self.hypernetwork(feedback_ids, feedback_mask)
                queries = [sample["query"]] + list(sample["related_queries"])
                loss = sum(
                    kl_distillation_loss(
                        self.target.student_forward(q, lora),
                        self.target.teacher_forward(q, sample["feedback"]),
                    )
                    for q in queries
                ) / len(queries)
                losses.append(loss.item())
        return sum(losses) / max(1, len(losses))

    # ---------- main loop ----------

    def fit(self) -> dict:
        history: dict[str, list[float]] = {"train": [], "val": []}
        for epoch in range(self.start_epoch, self.config.max_epochs):
            pbar = tqdm(self.train_loader, desc=f"epoch {epoch}")
            for batch in pbar:
                loss = self.train_step(batch)
                history["train"].append(loss)
                pbar.set_postfix(loss=f"{loss:.4f}")

                if self.wandb:
                    self.wandb.log({"train/loss": loss}, step=self.global_step)

                if self.global_step % self.config.save_every_steps == 0:
                    # Local-only save for in-instance resume; no HF upload (too slow).
                    self._save(Path(self.config.checkpoint_dir) / "latest.pt", epoch)

                if self.global_step % self.config.eval_every_steps == 0:
                    val_loss = self.validate()
                    history["val"].append(val_loss)
                    if self.wandb:
                        self.wandb.log({"val/loss": val_loss}, step=self.global_step)
                    self._maybe_checkpoint(val_loss, epoch)
                    if self.patience >= self.config.early_stopping_patience:
                        return history

            val_loss = self.validate()
            history["val"].append(val_loss)
            if self.wandb:
                self.wandb.log({"val/loss": val_loss, "epoch": epoch}, step=self.global_step)
            self._maybe_checkpoint(val_loss, epoch)
            if self.patience >= self.config.early_stopping_patience:
                return history
        return history

    def _trainable_state(self) -> dict:
        # Only projection heads are trained; backbone is frozen and reloadable from HF.
        return {
            k: v for k, v in self.hypernetwork.state_dict().items()
            if not k.startswith("backbone.")
        }

    def _save(self, path: Path, epoch: int, val_loss: float | None = None) -> None:
        """Full local checkpoint: model + optimizer for exact resume."""
        torch.save(
            {
                "model": self._trainable_state(),
                "optimizer": self.optimizer.state_dict(),
                "step": self.global_step,
                "epoch": epoch,
                "best_val": self.best_val,
                "patience": self.patience,
                "val_loss": val_loss,
                "wandb_id": self.wandb.run.id if self.wandb else None,
            },
            path,
        )

    def _save_light(self, path: Path, epoch: int, val_loss: float | None = None) -> None:
        """Lightweight checkpoint for HF upload: no optimizer state."""
        torch.save(
            {
                "model": self._trainable_state(),
                "step": self.global_step,
                "epoch": epoch,
                "val_loss": val_loss,
            },
            path,
        )

    def _maybe_checkpoint(self, val_loss: float, epoch: int) -> None:
        if val_loss < self.best_val:
            self.best_val = val_loss
            self.patience = 0
            ckpt_dir = Path(self.config.checkpoint_dir)
            self._save(ckpt_dir / "best.pt", epoch, val_loss)
            light = ckpt_dir / "best_light.pt"
            self._save_light(light, epoch, val_loss)
            self._upload_checkpoint(light)
        else:
            self.patience += 1

    def _upload_checkpoint(self, path: Path) -> None:
        try:
            from huggingface_hub import HfApi
            import os
            repo_id = "james-kernel/feedback-to-lora-checkpoints"
            api = HfApi()
            api.create_repo(repo_id, repo_type="model", exist_ok=True)
            api.upload_file(
                path_or_fileobj=str(path),
                path_in_repo=path.name,
                repo_id=repo_id,
                repo_type="model",
            )
            print(f"[hub] checkpoint uploaded -> {repo_id}/{path.name} (step {self.global_step})")
        except Exception as e:
            print(f"[hub] upload failed (non-fatal): {e}")

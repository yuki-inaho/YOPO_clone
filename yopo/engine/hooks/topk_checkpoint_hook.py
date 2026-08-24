# Copyright (c) OpenMMLab. All rights reserved.
"""Keep the top-K checkpoints ranked by a validation metric.

This is a drop-in extension of mmengine's standard ``CheckpointHook``: it keeps
the normal periodic + best checkpoints AND additionally retains the K
checkpoints with the best (highest/lower per ``rule``) metric values, deleting
the worst-performing extra checkpoints as training proceeds.

The tracked ranking lives in ``runner.message_hub`` under the
``topk_ckpt_<key_indicator>`` key, so it survives resume across restarts.
"""
import os
from typing import List, Optional, Tuple

from mmengine.hooks import CheckpointHook
from mmengine.runner import Runner

from yopo.registry import HOOKS


@HOOKS.register_module()
class TopKCheckpointHook(CheckpointHook):
    """Save the checkpoints with the top-K validation scores.

    Args:
        topk (int): Number of best checkpoints to keep. Defaults to 3.
        key_indicator (str): Metric used for ranking, e.g. ``NOCSMetric`` keys
            reported by the evaluator. Defaults to 'NOCSMetric'.
        rule (str, optional): 'greater' or 'less' (or short forms 'max'/'min').
            How to compare the metric; defaults to 'greater'.
        interval (int): Checkpoint saving interval (epochs). Defaults to 1.
        by_epoch (bool): Whether ``interval`` is in epochs (vs iters).
            Defaults to True.
        save_optimizer (bool): Whether to save the optimizer state in the
            top-k checkpoints. Defaults to False (weights only).
    """

    def __init__(self,
                 topk: int = 3,
                 key_indicator: str = 'NOCSMetric',
                 rule: Optional[str] = None,
                 interval: int = 1,
                 by_epoch: bool = True,
                 save_optimizer: bool = False,
                 **kwargs):
        # In mmengine, when rule is None it defaults to 'greater' (via
        # greater_keys). CheckpointHook derives rule from key_indicator suffix
        # only when rule is not given. We keep it explicit below.
        if rule is None:
            rule = 'greater'
        # ``save_best`` handled natively below (ranked by key_indicator); pop
        # any value passed through the config dict to avoid duplicate kwargs.
        kwargs.pop('save_best', None)
        super().__init__(
            interval=interval,
            by_epoch=by_epoch,
            save_optimizer=save_optimizer,
            save_best=key_indicator,
            rule=rule,
            **kwargs)
        self.topk = topk
        self.key_indicator = key_indicator
        self.rule = rule
        self.save_optimizer = save_optimizer

    # ------------------------------------------------------------------
    def after_val_epoch(self, runner, metrics):
        """Standard best checkpoint logic + top-K pruning.

        Periodic ('every N epochs') saving is handled by the inherited
        ``after_train_epoch``; this only adds the best (inherited) and the
        top-K ranked pool.
        """
        if len(metrics) == 0:
            runner.logger.warning(
                '`metrics` is empty; skipping top-k checkpoint update.')
            return

        # Best checkpoint saving (inherited).
        self._save_best_checkpoint(runner, metrics)

        # Top-K ranking on the configured metric.
        if self.key_indicator not in metrics \
                or metrics[self.key_indicator] is None:
            runner.logger.warning(
                f'`{self.key_indicator}` not found in metrics '
                f'{list(metrics.keys())}; skipping top-k.')
            return
        score = float(metrics[self.key_indicator])
        self._update_topk(runner, score)

    def _is_better(self, a: float, b: float) -> bool:
        return a > b if self.rule in ('greater', 'max') else a < b

    def _update_topk(self, runner: Runner, score: float) -> None:
        """Save current model into the top-k pool and prune the worst."""
        if not self.file_backend.isdir(self.out_dir):
            self.file_backend.makedirs(self.out_dir)
        rank = runner.rank
        if rank != 0:
            return

        # Current step label.
        if self.by_epoch:
            step = runner.epoch + 1
            step_label = f'epoch_{step}'
        else:
            step = runner.iter + 1
            step_label = f'iter_{step}'

        ckpt_name = (f'topk_{step_label}_{self.rule}{score:.4f}.pth')
        # Avoid illegal filename chars (e.g. '/', or metric names with dots ok).
        ckpt_name = ckpt_name.replace('/', '_')
        ckpt_path = os.path.join(self.out_dir, ckpt_name)

        runner.save_checkpoint(
            self.out_dir,
            filename=ckpt_name,
            file_client_args=self.file_client_args,
            save_optimizer=self.save_optimizer,
            save_param_scheduler=False,
            meta={},
            by_epoch=False,
            backend_args=self.backend_args)

        # Maintain top-K list in message_hub (resume-friendly).
        hub_key = f'topk_ckpt_{self.key_indicator}'
        current = runner.message_hub.get_info(hub_key, default=[])
        entry = (score, ckpt_path)
        current = [e for e in current if e[1] != ckpt_path]
        current.append(entry)
        # Sort best-first.
        current.sort(key=lambda e: e[0], reverse=(self.rule in
                                                  ('greater', 'max')))
        # Prune beyond topk -> remove the worst files.
        while len(current) > self.topk:
            _, worst_path = current.pop()
            if os.path.isfile(worst_path):
                os.remove(worst_path)
                runner.logger.info(
                    f'[TopK] removed worst checkpoint: {worst_path}')
            else:
                runner.logger.warning(
                    f'[TopK] could not remove {worst_path} (not a file).')
        runner.message_hub.update_info(hub_key, current)
        runner.logger.info(f'[TopK] pool={len(current)}/{self.topk} '
                           f'score={score:.4f} {self.key_indicator}')

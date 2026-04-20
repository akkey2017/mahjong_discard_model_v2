"""
Utility functions for training and evaluation.
"""

import math
import sys
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, ReduceLROnPlateau
from tqdm import tqdm


def _amp_device_type(device):
    """Return a device-type string compatible with :mod:`torch.amp`.

    Accepts either a ``str`` (e.g. ``"cuda"``, ``"cuda:0"``, ``"mps"``) or a
    :class:`torch.device` instance. Indices and sub-device suffixes are
    stripped. Unsupported types fall back to ``"cpu"`` so autocast does not
    silently run on a device it cannot accelerate.
    """
    if isinstance(device, torch.device):
        kind = device.type
    else:
        kind = str(device).split(":", 1)[0]
    if kind in ("cuda", "cpu", "mps", "xpu"):
        return kind
    return "cpu"


class TopKAccuracy:
    """Calculate top-k accuracy metric."""
    
    def __init__(self, k=3):
        self.k = k
        self.correct = 0
        self.total = 0
    
    def update(self, preds, labels):
        """Update accuracy with new predictions."""
        _, top_k_preds = preds.topk(self.k, dim=1)
        self.correct += torch.any(top_k_preds == labels.view(-1, 1), dim=1).sum().item()
        self.total += labels.size(0)
    
    def compute(self):
        """Compute current accuracy."""
        return self.correct / self.total if self.total > 0 else 0.0
    
    def reset(self):
        """Reset counters."""
        self.correct = 0
        self.total = 0


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience=5, min_delta=0.0, mode='max'):
        """
        Args:
            patience: How many epochs to wait after last improvement
            min_delta: Minimum change to qualify as an improvement
            mode: 'max' for metrics to maximize, 'min' for metrics to minimize
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score):
        """Check if training should stop."""
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


class ModelCheckpoint:
    """Save model checkpoints during training.

    When ``model_type`` is provided, the checkpoint is a rich dict of the form::

        {
            "model_state": ...,
            "model_type": "...",          # e.g. 'vit_large'
            "config": {...},              # full run config (CLI args)
            "extra": {...},               # optional run-time info
        }

    This lets :func:`load_checkpoint` restore the correct architecture without
    relying on filename heuristics.

    When ``model_type`` is ``None`` (legacy callers that expect a plain
    ``state_dict`` via ``torch.load`` + ``load_state_dict``), the checkpoint
    is written as a plain ``state_dict`` for backwards compatibility.
    """

    def __init__(self, filepath, monitor='val_loss', mode='min',
                 save_best_only=True, model_type=None, config=None):
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.best_score = None
        self.model_type = model_type
        self.config = config or {}

    def _improved(self, score):
        if self.best_score is None:
            return True
        if self.mode == 'min':
            return score < self.best_score
        return score > self.best_score

    def __call__(self, model, metrics, extra=None):
        score = metrics.get(self.monitor)
        if score is None:
            return

        should_save = (not self.save_best_only) or self._improved(score)
        if should_save:
            self.best_score = score
            save_checkpoint(
                self.filepath,
                model,
                model_type=self.model_type,
                config=self.config,
                extra=extra or {"metrics": metrics},
            )
            tqdm.write(
                f"Model checkpoint saved to {self.filepath} "
                f"({self.monitor}={score:.4f})"
            )


def save_checkpoint(filepath, model, model_type=None, config=None, extra=None):
    """Save a model checkpoint.

    If ``model_type`` is provided, a rich dict (including architecture info) is
    saved. If ``model_type`` is ``None``, a plain ``state_dict`` is written to
    preserve backwards compatibility with callers that do
    ``torch.load(path)`` + ``model.load_state_dict(...)`` directly.
    """
    if model_type is None:
        torch.save(model.state_dict(), filepath)
        return
    payload = {
        "model_state": model.state_dict(),
        "model_type": model_type,
        "config": config or {},
        "extra": extra or {},
    }
    torch.save(payload, filepath)


def load_checkpoint(filepath, map_location="cpu"):
    """Load a rich checkpoint dict.

    Falls back to plain state_dict (for backwards-compat with old files) by
    returning ``{"model_state": <state_dict>, "model_type": None, "config": {}}``.
    """
    # weights_only=True prevents arbitrary code execution from untrusted files.
    # Our checkpoints contain only tensors and primitive Python types (str/int/float/list/dict).
    obj = torch.load(filepath, map_location=map_location, weights_only=True)
    if isinstance(obj, dict) and "model_state" in obj:
        return obj
    return {"model_state": obj, "model_type": None, "config": {}, "extra": {}}


def get_optimizer(model, optimizer_name='adamw', lr=1e-4, weight_decay=1e-2):
    """
    Get optimizer for the model.
    
    Args:
        model: The neural network model
        optimizer_name: Name of the optimizer ('adam', 'adamw', 'sgd')
        lr: Learning rate
        weight_decay: Weight decay factor
    
    Returns:
        PyTorch optimizer
    """
    if optimizer_name.lower() == 'adam':
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def get_scheduler(optimizer, scheduler_name='cosine', **kwargs):
    """
    Get learning rate scheduler.

    Args:
        optimizer: PyTorch optimizer
        scheduler_name: Name of the scheduler
            ('cosine', 'warmup_cosine', 'plateau', 'none')
        **kwargs: Additional arguments for the scheduler.
            For 'cosine':
                - T_max (int): total number of epochs
                - eta_min (float): absolute final learning rate (default 1e-6)
            For 'warmup_cosine':
                - T_max (int): total number of epochs
                - warmup_epochs (int): number of warmup epochs (default
                  ``min(5, max(1, T_max // 5))``)
                - eta_min_ratio (float): floor of the LR multiplier at the end
                  of the cosine decay, expressed as a fraction of the base LR
                  (default 0.01 -> final LR = 1% of base LR). This is a
                  *ratio*, not an absolute LR.

    Returns:
        PyTorch learning rate scheduler or None
    """
    name = scheduler_name.lower()
    if name == 'cosine':
        T_max = kwargs.get('T_max', 10)
        eta_min = kwargs.get('eta_min', 1e-6)
        return CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    elif name == 'warmup_cosine':
        T_max = kwargs.get('T_max', 10)
        warmup_epochs = kwargs.get('warmup_epochs', min(5, max(1, T_max // 5)))
        eta_min_ratio = kwargs.get('eta_min_ratio', 0.01)

        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return float(epoch + 1) / float(max(1, warmup_epochs))
            progress = (epoch - warmup_epochs) / max(1, T_max - warmup_epochs)
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return eta_min_ratio + (1.0 - eta_min_ratio) * cosine

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        # Training loops call scheduler.step() AFTER each epoch, so ensure
        # epoch 0 actually trains at the warmup-start LR (= base_lr/warmup_epochs)
        # rather than the optimizer's base LR. Keep the scheduler state aligned
        # with that manual initialization so the first post-epoch step advances
        # to lr_lambda(1) instead of repeating lr_lambda(0).
        initial_multiplier = lr_lambda(0)
        for base_lr, group in zip(scheduler.base_lrs, optimizer.param_groups):
            group['lr'] = base_lr * initial_multiplier
        scheduler.last_epoch = 0
        return scheduler
    elif name == 'plateau':
        mode = kwargs.get('mode', 'max')
        factor = kwargs.get('factor', 0.5)
        patience = kwargs.get('patience', 3)
        return ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience)
    elif name == 'none':
        return None
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")


def train_one_epoch(
    model,
    train_loader,
    loss_fn,
    optimizer,
    device,
    max_grad_norm=None,
    scaler=None,
    use_amp=False,
    accumulation_steps=1,
    ema=None,
):
    """Train the model for one epoch.

    Args:
        model: Neural network model
        train_loader: DataLoader for training data
        loss_fn: Loss function
        optimizer: Optimizer
        device: Device to train on
        max_grad_norm: Maximum gradient norm for clipping (None to disable)
        scaler: ``torch.amp.GradScaler`` instance for mixed-precision training
        use_amp: Whether to run forward under ``torch.amp.autocast``
        accumulation_steps: Number of micro-batches per optimizer step
        ema: Optional :class:`ModelEMA` instance; updated after every optimizer
            step (i.e. every ``accumulation_steps`` micro-batches) so the
            effective decay matches what the constructor's ``decay`` value implies.

    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    accumulation_steps = max(1, accumulation_steps)
    microbatches_in_window = 0  # actual micro-batches accumulated since last step
    samples_in_window = 0       # total samples whose loss was accumulated

    amp_device_type = _amp_device_type(device)

    def _flush_gradients(sample_count):
        # Normalize the accumulated gradients by the total number of samples
        # in the window so each sample contributes equally regardless of
        # batch size (e.g. the last micro-batch may be smaller than the rest).
        # This is equivalent to running a single forward/backward over the
        # concatenated window with ``reduction="mean"``.
        if sample_count > 1:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.div_(sample_count)
        if max_grad_norm is not None:
            if scaler is not None and use_amp:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        if scaler is not None and use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if ema is not None:
            ema.update(model)

    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(train_loader, desc="Training", leave=True, file=sys.stderr,
                dynamic_ncols=True, mininterval=0.1, unit="batch")
    for step, (xb, yb, _) in enumerate(pbar):
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=amp_device_type, enabled=use_amp):
            out = model(xb)
            loss = loss_fn(out, yb)

        batch_size = yb.size(0)
        # Scale the mean loss back to a sum so the accumulated gradients can
        # be divided by ``samples_in_window`` at flush time.
        scaled_loss = loss * batch_size
        if scaler is not None and use_amp:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        microbatches_in_window += 1
        samples_in_window += batch_size

        if (step + 1) % accumulation_steps == 0:
            _flush_gradients(samples_in_window)
            microbatches_in_window = 0
            samples_in_window = 0

        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix(loss=f"{total_loss / num_batches:.4f}")

    # Flush any gradients from a partial final accumulation window.
    if microbatches_in_window > 0:
        _flush_gradients(samples_in_window)

    return total_loss / num_batches if num_batches > 0 else 0.0


def evaluate(model, val_loader, loss_fn, device, metrics=None, use_amp=False):
    """
    Evaluate the model.

    Args:
        model: Neural network model
        val_loader: DataLoader for validation data
        loss_fn: Loss function
        device: Device to evaluate on
        metrics: Dictionary of metric objects to update (e.g., TopKAccuracy)
        use_amp: Whether to run forward under ``torch.amp.autocast``

    Returns:
        Dictionary containing evaluation results
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    amp_device_type = _amp_device_type(device)

    # Reset metrics
    if metrics:
        for metric in metrics.values():
            if hasattr(metric, 'reset'):
                metric.reset()
    
    pbar = tqdm(val_loader, desc="Evaluating", leave=True, file=sys.stderr, dynamic_ncols=True, mininterval=0.1, unit="batch")
    with torch.no_grad():
        for xb, yb, _ in pbar:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)

            with torch.amp.autocast(device_type=amp_device_type, enabled=use_amp):
                out = model(xb)
                loss = loss_fn(out, yb)
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update metrics
            if metrics:
                for metric in metrics.values():
                    if hasattr(metric, 'update'):
                        metric.update(out, yb)
            
            # Update progress bar with current loss
            if num_batches > 0:
                avg_loss = total_loss / num_batches
                pbar.set_postfix(loss=f"{avg_loss:.4f}")
    
    results = {'loss': total_loss / num_batches if num_batches > 0 else 0.0}
    
    # Compute metrics
    if metrics:
        for name, metric in metrics.items():
            if hasattr(metric, 'compute'):
                results[name] = metric.compute()
    
    return results


def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model, input_shape=(1, 380, 4, 9)):
    """
    Print a summary of the model architecture.

    Args:
        model: Neural network model
        input_shape: Shape of the input tensor
    """
    print(f"\n{'='*60}")
    print(f"Model Summary")
    print(f"{'='*60}")
    print(f"Total parameters: {count_parameters(model):,}")
    print(f"Input shape: {input_shape}")

    try:
        device = next(model.parameters()).device
        dummy_input = torch.randn(*input_shape).to(device)
        with torch.no_grad():
            output = model(dummy_input)
        if isinstance(output, dict):
            shapes = {k: tuple(v.shape) for k, v in output.items()}
            print(f"Output heads: {shapes}")
        else:
            print(f"Output shape: {tuple(output.shape)}")
    except Exception as e:
        print(f"Could not determine output shape: {e}")

    print(f"{'='*60}\n")


class ModelEMA:
    """Exponential Moving Average of model parameters.

    Keeps a shadow copy of ``model`` and updates it as
    ``v_t = decay * v_{t-1} + (1 - decay) * param_t`` after each optimizer step.
    Access the EMA model via the ``.ema`` attribute for evaluation.
    """

    def __init__(self, model, decay=0.9999):
        import copy
        self.decay = decay
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(msd[k].detach(), alpha=1.0 - self.decay)
            else:
                v.copy_(msd[k])


def train_one_epoch_multitask(
    model,
    train_loader,
    loss_fns,
    optimizer,
    device,
    task_weights=None,
    max_grad_norm=None,
    scaler=None,
    use_amp=False,
    accumulation_steps=1,
    ema=None,
):
    """Multi-task training epoch.

    Each batch yields (x, y, action_type). Samples sharing the same
    ``action_type`` within a batch are routed to the matching head in
    ``model.heads`` and contribute to the head-specific loss.

    Args:
        model: :class:`MultiTaskDiscardModel` (returns dict of logits).
        train_loader: yields (x, y, action_type_list_or_tensor).
        loss_fns: dict mapping head name -> loss function.
        task_weights: dict mapping head name -> scalar weight (default 1.0).
        ema: Optional :class:`ModelEMA` instance; updated after every optimizer step.
        Others: same as :func:`train_one_epoch`.
    """
    model.train()
    task_weights = task_weights or {}
    accumulation_steps = max(1, accumulation_steps)
    microbatches_in_window = 0  # actual micro-batches accumulated since last step
    samples_in_window = 0       # total samples whose loss was accumulated
    amp_device_type = _amp_device_type(device)

    def _flush_gradients(sample_count):
        # Normalize accumulated gradients by the total number of contributing
        # samples in the window.  Per-task losses are computed as sums (see
        # below) so dividing by ``sample_count`` yields a sample-weighted
        # average that is invariant to both batch size and per-task subset size.
        if sample_count > 1:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.div_(sample_count)
        if max_grad_norm is not None:
            if scaler is not None and use_amp:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        if scaler is not None and use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if ema is not None:
            ema.update(model)

    total_loss = 0.0
    num_batches = 0
    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(train_loader, desc="Training", leave=True, file=sys.stderr,
                dynamic_ncols=True, mininterval=0.1, unit="batch")

    for step, (xb, yb, actions) in enumerate(pbar):
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        # actions may be a list[str] or tensor; treat as list of strings
        if torch.is_tensor(actions):
            actions = [str(a) for a in actions.tolist()]

        # Collect sample indices per action type
        per_task_idx = {}
        for i, a in enumerate(actions):
            per_task_idx.setdefault(a, []).append(i)

        active_tasks = [t for t in per_task_idx if t in model.heads]
        if not active_tasks:
            continue

        with torch.amp.autocast(device_type=amp_device_type, enabled=use_amp):
            logits_all = model(xb, head_names=active_tasks)
            loss_total = None
            micro_samples = 0
            for task in active_tasks:
                idx = torch.tensor(per_task_idx[task], device=device, dtype=torch.long)
                logits = logits_all[task].index_select(0, idx)
                targets = yb.index_select(0, idx)
                lfn = loss_fns.get(task, loss_fns.get("_default"))
                if lfn is None:
                    continue
                weight = float(task_weights.get(task, 1.0))
                subset_size = idx.numel()
                # Multiply by subset_size to convert the head's mean-reduced
                # loss back into a sum.  Combined with the sample-weighted
                # flush below, each sample contributes equally regardless of
                # how it splits across heads or batches.
                l = lfn(logits, targets) * weight * subset_size
                loss_total = l if loss_total is None else loss_total + l
                micro_samples += subset_size

            if loss_total is None:
                continue

        if scaler is not None and use_amp:
            scaler.scale(loss_total).backward()
        else:
            loss_total.backward()
        microbatches_in_window += 1
        samples_in_window += micro_samples

        if (step + 1) % accumulation_steps == 0:
            _flush_gradients(samples_in_window)
            microbatches_in_window = 0
            samples_in_window = 0

        # Report an average per-sample loss for progress/logging (the stored
        # ``loss_total`` is a sum, so normalize it here).
        total_loss += (loss_total.item() / max(1, micro_samples))
        num_batches += 1
        pbar.set_postfix(loss=f"{total_loss / max(1, num_batches):.4f}")

    # Flush any gradients from a partial final accumulation window.
    if microbatches_in_window > 0:
        _flush_gradients(samples_in_window)

    return total_loss / num_batches if num_batches > 0 else 0.0


def evaluate_multitask(model, val_loader, loss_fns, device, task_weights=None, use_amp=False):
    """Multi-task evaluation. Returns per-task loss and top-1 accuracy.

    Per-task losses are sample-weighted: each batch's per-task loss (a mean
    over that task's subset of the batch) is multiplied by the number of
    samples routed to that head before being aggregated, so tasks with
    variable per-batch subset sizes are not skewed by batch boundaries.
    """
    model.eval()
    task_weights = task_weights or {}
    amp_device_type = _amp_device_type(device)

    per_task_correct = {}
    per_task_total = {}
    per_task_loss_sum = {}        # sample-weighted running sum
    per_task_loss_samples = {}    # number of samples contributing to the sum

    pbar = tqdm(val_loader, desc="Evaluating", leave=True, file=sys.stderr,
                dynamic_ncols=True, mininterval=0.1, unit="batch")
    with torch.no_grad():
        for xb, yb, actions in pbar:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            if torch.is_tensor(actions):
                actions = [str(a) for a in actions.tolist()]

            per_task_idx = {}
            for i, a in enumerate(actions):
                per_task_idx.setdefault(a, []).append(i)

            active_tasks = [t for t in per_task_idx if t in model.heads]
            if not active_tasks:
                continue

            with torch.amp.autocast(device_type=amp_device_type, enabled=use_amp):
                logits_all = model(xb, head_names=active_tasks)
                for task in active_tasks:
                    idx = torch.tensor(per_task_idx[task], device=device, dtype=torch.long)
                    logits = logits_all[task].index_select(0, idx)
                    targets = yb.index_select(0, idx)
                    n = int(targets.numel())
                    lfn = loss_fns.get(task, loss_fns.get("_default"))
                    if lfn is not None and n > 0:
                        weight = float(task_weights.get(task, 1.0))
                        l = lfn(logits, targets).item() * weight
                        # l is a mean over this subset; multiply by n to
                        # accumulate a sample-weighted sum.
                        per_task_loss_sum[task] = per_task_loss_sum.get(task, 0.0) + l * n
                        per_task_loss_samples[task] = per_task_loss_samples.get(task, 0) + n
                    pred = logits.argmax(dim=-1)
                    per_task_correct[task] = per_task_correct.get(task, 0) + int((pred == targets).sum())
                    per_task_total[task] = per_task_total.get(task, 0) + n

    results = {}
    total_samples = sum(per_task_total.values()) or 1
    overall_correct = sum(per_task_correct.values())
    results["top1_acc"] = overall_correct / total_samples
    for task, total in per_task_total.items():
        acc = per_task_correct[task] / total if total else 0.0
        samples_for_loss = per_task_loss_samples.get(task, 0)
        avg_loss = (per_task_loss_sum.get(task, 0.0) / samples_for_loss) if samples_for_loss else 0.0
        results[f"{task}_acc"] = acc
        results[f"{task}_loss"] = avg_loss
        results[f"{task}_total"] = total
    overall_loss_samples = sum(per_task_loss_samples.values())
    results["loss"] = (sum(per_task_loss_sum.values()) / overall_loss_samples) if overall_loss_samples else 0.0
    return results

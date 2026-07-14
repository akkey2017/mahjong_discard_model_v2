# ViT checkpoint compatibility

The step-based ViT stack writes `vit-training-checkpoint-v1` checkpoints. A
resume checkpoint must match all of the following exactly:

- `ViTConfig`
- feature schema version
- multi-task target schema version
- prepared dataset manifest SHA-256
- maximum/warmup steps and gradient accumulation

New checkpoints contain training weights, AdamW state, step scheduler state,
AMP scaler state, EMA weights, RNG state, global sample/step counters, and the
per-worker stream position. They can be passed to `scripts/train_vit.py
--resume`.

Legacy policy:

- `vit_multitask_large` from `advanced_training` is shape- and key-compatible
  with the new `vit_large`. It can be loaded strictly with `--init-checkpoint`.
- Legacy single-head `vit_large` loads its backbone, shared normalization, and
  discard classifier. The other four heads remain newly initialized.
- Legacy checkpoints are initialization sources only. They cannot be used for
  exact resume because they do not contain the new step scheduler, target
  schema, dataset lineage, or stream position.
- CoAtNet and ResNet checkpoints are unsupported by the ViT-only stack.

Incompatible schemas fail before model or optimizer training resumes. Removing
or bypassing the schema check is not supported.

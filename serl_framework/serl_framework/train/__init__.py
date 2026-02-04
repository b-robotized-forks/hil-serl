"""Training entry points and helpers for HIL-SERL.

This subpackage bundles the executable training scripts (run as modules, e.g.
``python -m serl_framework.train.train_rlpd``) together with their shared
helpers. Nothing is imported here so that ``import serl_framework`` stays free
of heavy dependencies like jax; the scripts pull those in only when executed.
"""

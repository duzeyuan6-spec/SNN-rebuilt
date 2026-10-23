# Upload validation

- Python compilation: passed.
- Repository reconstruction tests: 14 passed.
- Metric structure-loss tests: 2 passed.
- Reconstruction evaluation tests: 6 passed.
- MR .10/.25/.50 shape, gradients, shared Phi and OCTUF .25 compatibility: passed using upstream OCTUF source.
- Portability entry points: unified_hardware.py and analyze_activity.py --help passed.
- Core source files copied without model/training edits; SOURCE_MANIFEST records original source-byte digests (Git line-ending normalization may change checkout-byte digests).
- Hardware/model loading behavior was previously exercised on the server; the packaged portable wrappers have not rerun the full GPU training/benchmark here.
- New staged code scanned for credentials, private server addresses and local user paths; none found. Datasets, checkpoints, local runtime dependencies and raw logs excluded.

Root migration: replaced the legacy working tree with this frozen implementation; repeated compilation, 22 unit tests, all three MR compatibility checks and hardware CLI loading successfully. Model/training implementations unchanged from the preceding release.

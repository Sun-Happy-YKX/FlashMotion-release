# Recorded Mamba Python-interface repair

`selective_scan_interface.observed.py` is the exact 20,311-byte provider file
recorded in all three [runtime inventories](../../docs/runtime/), SHA-256
`2260a1486cea11fff63ba69d584175864f2687354b722fa9c16820de6a45158f`.
It was read back without importing a model or modifying the original runtime.

The upstream base is Mamba commit
[`10b5d6358f27966f6a40e4bf0baa17a460688128`](https://github.com/state-spaces/mamba/blob/10b5d6358f27966f6a40e4bf0baa17a460688128/mamba_ssm/ops/selective_scan_interface.py),
whose 20,025-byte interface has SHA-256
`543edd077f11c80cccd5c5e2f926420d9105fd238d9e8960e53660cfa234dce8`.
The adjacent unified diff records the only difference between these two files:
the existing local repair changes one `torch.addmm(..., out=...)` call to the
out-of-place form and adds its explanatory comment. It is not an upstream
release change, a replacement CUDA kernel, or proof of JVP/metric correctness.
No claim is made that all other files in the original provider worktree were
clean merely because its Git HEAD was recorded.

The Mamba source retains its copyright notice for Tri Dao and Albert Gu and is
distributed under the included upstream [Apache-2.0 license](LICENSE.mamba.Apache-2.0.txt).
The recorded modification is marked in the source comment and this notice.
The [explicit hash-gated installer](../../tools/patch_mamba_interface.py) changes
only the matching installed Python file in a newly created task conda prefix,
keeping an upstream-byte backup. It never imports native providers or installs
packages. Follow [the environment guide](../../docs/environment.md).

# Origins

Every file, and where it comes from. "New" means written for this repository. The derived files keep the notice of
tenstorrent/tt-metal (Apache-2.0) in their head.

| File | Origin |
|---|---|
| `kernels/gdn_compute.cpp` | Adapted from tenstorrent/tt-metal v0.81.1, `ttnn/cpp/ttnn/operations/transformer/chunk_gated_delta_rule/device/kernels/compute/chunk_gated_delta_rule.cpp` (the "mono" program of `chunk_gated_delta_rule`), itself derived from flash-linear-attention's `naive_chunk_gated_delta_rule`. Changes: several heads per core; the q/k L2 norm and the q scale in the kernel; the WY inverse per 32 x 32 diagonal block; the gated RMSNorm of the output; the state update at float32. |
| `kernels/gdn_reader.cpp` | Adapted from tt-metal v0.81.1, `.../device/kernels/dataflow/reader_chunk_gated_delta_rule.cpp` and `chunk_gdn_gate_read.hpp`. Changes: flat token-major or head-major inputs with a column offset; several heads per core. |
| `kernels/gdn_writer.cpp` | Adapted from tt-metal v0.81.1, `.../device/kernels/dataflow/writer_chunk_gated_delta_rule.cpp`. Changes: flat or head-major output; several heads per core. |
| `kernels/gdn_ops.h` | The tile helpers of `gdn_compute.cpp` (above, so derived from tt-metal), shared with the verify kernel. |
| `kernels/gdn_compat.h` | New: one name for each compute-API call the kernels make, and partial-tile SFPU helpers on tt-metal's LLK entry points. |
| `kernels/gdn_verify_compute.cpp`, `gdn_verify_reader.cpp`, `gdn_verify_writer.cpp` | New. The compute kernel uses `gdn_ops.h`. |
| `kernels/gdn_conv_compute.cpp`, `gdn_conv_reader.cpp`, `gdn_conv_writer.cpp` | New. They compute what upstream's `_causal_conv1d_decode_t1` and `_causal_conv1d_fir` compute (tt-metal v0.81.1, `models/experimental/gated_attention_gated_deltanet/tt/ttnn_gated_deltanet.py`). |
| `kernels/gdn_pconv_*.cpp`, `kernels/gdn_pgates_*.cpp` | New. Bit for bit with upstream's `_causal_conv1d_fir` and the gate ops of its prefill, by using the same SFPU operations in the same order. |
| `kernels/gdn_commit_reader.cpp`, `gdn_commit_writer.cpp` | New. |
| `gdn_scan.py` | New. Its program layout follows upstream's `chunk_gdn_mono_program_factory.cpp` (tt-metal v0.81.1). `chunk_gated_delta_rule_seq_adapter` has the interface of upstream's adapter of the same name. |
| `gdn_verify.py`, `gdn_decode.py`, `gdn_conv.py`, `gdn_pconv.py`, `gdn_commit.py`, `mamba2.py` | New. `gdn_pconv.stock` and `stock_gates` call upstream's model code, which is not included here. |
| `tests/*.py` | New. |


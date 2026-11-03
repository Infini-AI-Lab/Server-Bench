# Mixture of experts

*Applies to: `moe_router`, `moe_grouped_gemm`. Not exercised by a dense model
like Qwen3-4B; here because the operator inventory covers architectures the
benchmark's later tasks reach.*

An MoE layer routes each token to `k` of `E` experts and runs a GEMM per expert
over the tokens that chose it. The kernel problem is that the per-expert token
counts are data-dependent and change every step.

## The shape of the work

1. **Route.** Score the experts per token, take top-`k`, normalise the weights.
2. **Permute.** Sort tokens by expert so each expert's rows are contiguous.
3. **Grouped GEMM.** One GEMM per expert over a variable number of rows.
4. **Unpermute and combine.** Scatter back, weighted by the routing weights.

The permutation is not overhead to be optimised away; it is what turns `E` tiny
irregular GEMMs into one grouped GEMM whose tiles are all full.

## Where it goes wrong

- **Load imbalance.** A popular expert gets far more tokens than the others and
  its GEMM decides the step. The grouped kernel should schedule tiles across all
  experts from one queue rather than one grid per expert.
- **The empty expert.** An expert chosen by no token this step has zero rows. A
  kernel that launches a grid of zero, or divides by the count, fails here — and
  only sometimes, which makes it a bad bug to find late.
- **Top-k weight normalisation.** Normalise before or after the expert
  computation, consistently with the reference. Getting it backwards is a
  small, systematic error that looks like noise.
- **Capacity dropping.** If the implementation caps tokens per expert, tokens
  beyond the cap are *dropped*, and that is a semantic change to the model, not
  an optimisation. It must be declared, not discovered.

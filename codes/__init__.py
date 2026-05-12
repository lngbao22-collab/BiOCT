# Package initializer for codes
# Ensure sparse tensor invariant checks are explicitly configured as early as possible
import torch
import warnings

# Explicitly opt out of sparse tensor invariant checks to silence implicit warnings
try:
    torch.sparse.check_sparse_tensor_invariants(False)
except Exception:
    pass

# Also filter matching UserWarnings (best-effort)
warnings.filterwarnings("ignore", message=".*Sparse invariant checks are implicitly disabled.*", category=UserWarning)

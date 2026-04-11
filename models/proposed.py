# Shim — all logic lives in proposed2.py.
# Importing from here gives ProposedModelV2 (the current default model).
from .proposed2 import (  # noqa: F401
    ProposedModel,
    ProposedModelV2,
    adapt_checkpoint_to_temporal,
    build_proposed,
    build_proposed2,
)

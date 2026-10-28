"""Alpha track: one pinned open base, adapted on a released internal dataset.

This package is the only place a published checkpoint may be loaded, and only
from a hash-checked snapshot already on disk. The company owns the adapter.
A change is a new dataset version, a new run, and a new human gate. The
from-scratch track in ``training/model.py`` stays randomly initialised, and
alpha answers are not training data for that track.
"""

from internal_ai.training.alpha.pin import FALLBACK, PRIMARY, AlphaRecipe

__all__ = ["FALLBACK", "PRIMARY", "AlphaRecipe"]

"""Training text is a dataset the company has released."""

from __future__ import annotations

from internal_ai.domain import DatasetVersion


def require_released_dataset(dataset: DatasetVersion | None) -> DatasetVersion:
    """Refuse a missing or unreleased dataset.

    An improvement is a new version, a new run, and a new human gate. Alpha
    output is not a dataset, and it is not training data for the from-scratch
    track.
    """
    if dataset is None:
        raise KeyError("unknown dataset")
    if not dataset.released:
        raise PermissionError(
            "a model is built from a released dataset; improve the model with a new version"
        )
    if dataset.stats.get("training_data") is False:
        raise PermissionError(
            "this set was exported from feedback and is not training data until its release"
        )
    return dataset

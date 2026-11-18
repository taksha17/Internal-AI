"""Which runtime class a job may name, and what that name isolates.

gVisor is the default on a cluster: a missing RuntimeClass leaves the pod
pending instead of running it on the host kernel. Kata and Firecracker are
the other classes this install will write into a manifest or a container
command. A name the table does not list is refused, because a free string
is how a job would ask for the host runtime.
"""

from __future__ import annotations

from dataclasses import dataclass


class RuntimeRefused(ValueError):
    """The runtime class is not one this install will place a job on."""


@dataclass(frozen=True, slots=True)
class Runtime:
    name: str
    handler: str
    isolation: str


RUNTIMES: tuple[Runtime, ...] = (
    Runtime(name="gvisor", handler="runsc", isolation="user-space kernel"),
    Runtime(name="kata", handler="kata-runtime", isolation="virtual machine"),
    Runtime(name="firecracker", handler="firecracker", isolation="microvm"),
)

_BY_NAME = {runtime.name: runtime for runtime in RUNTIMES}


def runtime_for(name: str) -> Runtime:
    found = _BY_NAME.get((name or "").strip())
    if found is None:
        known = ", ".join(runtime.name for runtime in RUNTIMES)
        raise RuntimeRefused(f"unknown runtime class {name!r}; known classes are {known}")
    return found


def described() -> list[dict[str, str]]:
    """Rows for the compute page, in table order."""
    return [
        {"name": runtime.name, "handler": runtime.handler, "isolation": runtime.isolation}
        for runtime in RUNTIMES
    ]

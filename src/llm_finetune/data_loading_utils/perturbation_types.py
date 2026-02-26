"""perturbation_types.py.

Perturbation taxonomy enum, convenience sets, and per-file load configuration.

These are leaf-level definitions with no internal dependencies, so they can be
imported from anywhere without risking circular imports.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


# ---------------------------------------------------------------------------
# Perturbation type taxonomy
# ---------------------------------------------------------------------------


class PerturbationType(Enum):
    """Perturbation types aligned with PERTURBATIONS.md naming."""

    ORIGINAL_NORMALIZED = (
        "original_normalized"  # normalized_claim key only (no perturbation)
    )
    PERTURB_TYPE1 = "perturb_type1"  # Structure-preserving
    PERTURB_TYPE2 = "perturb_type2"  # Value-modifying
    PERTURB_TYPE3_RANDOM = "perturb_type3_random"  # Obfuscation: random
    PERTURB_TYPE3_MASKED = "perturb_type3_masked"  # Obfuscation: masked
    PERTURB_TYPE3_REMOVED = "perturb_type3_removed"  # Obfuscation: removed


# Convenience sets
STRUCTURE_PRESERVING_TYPES = {PerturbationType.PERTURB_TYPE1}
VALUE_MODIFYING_TYPES = {PerturbationType.PERTURB_TYPE2}
OBFUSCATION_TYPES = {
    PerturbationType.PERTURB_TYPE3_RANDOM,
    PerturbationType.PERTURB_TYPE3_MASKED,
    PerturbationType.PERTURB_TYPE3_REMOVED,
}
ALL_PERTURB_TYPES = (
    STRUCTURE_PRESERVING_TYPES | VALUE_MODIFYING_TYPES | OBFUSCATION_TYPES
)


# ---------------------------------------------------------------------------
# File load configuration
# ---------------------------------------------------------------------------


@dataclass
class FileLoadConfig:
    """Configuration for loading a single perturbation JSON file.

    Args:
        path: Path to the JSON file.
        perturb_types: Which perturbation types to extract from this file.
            ``None`` (the default) loads every perturbation key present.
    """

    path: str
    perturb_types: Optional[List[PerturbationType]] = None

    def __repr__(self) -> str:
        types = (
            "all"
            if self.perturb_types is None
            else [t.value for t in self.perturb_types]
        )
        return f"FileLoadConfig(path={self.path!r}, perturb_types={types})"


def parse_file_config(config_str: str) -> FileLoadConfig:
    """Parse a CLI-style file config string into a :class:`FileLoadConfig`.

    Format: ``path/to/file.json[:type1,type2,...]``

    Shorthand type names are accepted (``type1`` → ``perturb_type1``).
    If no types are specified after the colon, all types are loaded.

    Examples::

        parse_file_config("approx_True.json")
        # → FileLoadConfig(path="approx_True.json", perturb_types=None)

        parse_file_config("approx_True.json:type1,type2")
        # → FileLoadConfig(path="approx_True.json",
        #     perturb_types=[PERTURB_TYPE1, PERTURB_TYPE2])

        parse_file_config("obfusc_False.json:type3_random,type3_masked")
        # → FileLoadConfig(path="obfusc_False.json",
        #     perturb_types=[PERTURB_TYPE3_RANDOM, PERTURB_TYPE3_MASKED])
    """
    if ":" in config_str:
        path, types_str = config_str.rsplit(":", 1)
        type_names = [t.strip() for t in types_str.split(",") if t.strip()]
        if not type_names:
            return FileLoadConfig(path=path)
        perturb_types: List[PerturbationType] = []
        for name in type_names:
            full_name = (
                name if name.startswith("perturb_") else f"perturb_{name}"
            )
            try:
                perturb_types.append(PerturbationType(full_name))
            except ValueError:
                raise ValueError(
                    f"Unknown perturbation type: {name!r}. "
                    f"Valid types: {[t.value for t in PerturbationType if t != PerturbationType.ORIGINAL_NORMALIZED]}"
                )
        return FileLoadConfig(path=path, perturb_types=perturb_types)
    return FileLoadConfig(path=config_str)

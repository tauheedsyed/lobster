"""
Construct modality adapters by their registered name.

This is the single place that maps an adapter name (for example
``"transcriptomics_single_cell"``) to a configured adapter instance.
``DataManagerV2`` registers its default adapters through it, and exported
notebooks load their input through :func:`load_with_adapter`, so a session and
the notebook generated from it build the same adapter with the same
configuration and cannot drift apart.

Resolution order matches the data manager's registration order: an adapter
factory published under the ``lobster.adapters`` entry-point group wins; the
built-in adapters below are the fallback.
"""

import logging
from typing import Any, Callable, Dict

logger = logging.getLogger(__name__)

__all__ = [
    "builtin_adapter_factories",
    "create_adapter",
    "create_default_adapters",
    "entry_point_adapter_factories",
    "load_with_adapter",
]


def entry_point_adapter_factories() -> Dict[str, Callable[[], Any]]:
    """Adapter factories published under the ``lobster.adapters`` entry points."""
    try:
        from lobster.core.component_registry import component_registry

        return dict(component_registry.list_adapters())
    except Exception as e:
        logger.debug(f"Entry point adapter discovery skipped: {e}")
        return {}


def builtin_adapter_factories() -> Dict[str, Callable[[], Any]]:
    """Built-in adapter factories whose dependencies are importable, in order."""
    factories: Dict[str, Callable[[], Any]] = {}

    # Transcriptomics adapters
    try:
        from lobster.core.adapters.transcriptomics_adapter import (
            TranscriptomicsAdapter,
        )
    except ImportError:
        logger.debug("Transcriptomics adapters not available")
    else:
        factories["transcriptomics_single_cell"] = lambda: TranscriptomicsAdapter(
            data_type="single_cell", strict_validation=False
        )
        factories["transcriptomics_bulk"] = lambda: TranscriptomicsAdapter(
            data_type="bulk", strict_validation=False
        )

    # Proteomics adapters (may not be available in every distribution)
    try:
        from lobster.core.adapters.proteomics_adapter import ProteomicsAdapter
    except ImportError:
        logger.debug("Proteomics adapters not available")
    else:
        factories["proteomics_ms"] = lambda: ProteomicsAdapter(
            data_type="mass_spectrometry", strict_validation=False
        )
        factories["proteomics_affinity"] = lambda: ProteomicsAdapter(
            data_type="affinity", strict_validation=False
        )

    # Genomics adapters (optional; registered together or not at all)
    try:
        from lobster.core.adapters.genomics.plink_adapter import PLINKAdapter
        from lobster.core.adapters.genomics.vcf_adapter import VCFAdapter
    except ImportError:
        logger.debug("Genomics adapters not available")
    else:
        factories["genomics_wgs"] = lambda: VCFAdapter(strict_validation=False)
        factories["genomics_snp_array"] = lambda: PLINKAdapter(strict_validation=False)

    return factories


def _create(
    name: str,
    entry_points: Dict[str, Callable[[], Any]],
    builtins: Dict[str, Callable[[], Any]],
) -> Any:
    factory = entry_points.get(name)
    if factory is not None:
        try:
            adapter = factory()
            logger.debug(f"Created adapter '{name}' from entry point")
            return adapter
        except Exception as e:
            if name not in builtins:
                raise
            logger.warning(f"Failed to load adapter '{name}' from entry point: {e}")
    if name not in builtins:
        available = sorted(set(entry_points) | set(builtins))
        raise ValueError(
            f"Unknown adapter '{name}'. Available adapters: {', '.join(available)}"
        )
    return builtins[name]()


def create_adapter(name: str) -> Any:
    """Create the adapter registered as ``name``.

    An entry-point factory is used first; if there is none, or it fails and a
    built-in adapter of the same name exists, the built-in one is created.

    Raises:
        ValueError: If no adapter is known under ``name``.
    """
    return _create(name, entry_point_adapter_factories(), builtin_adapter_factories())


def create_default_adapters() -> Dict[str, Any]:
    """Create every available adapter, keyed by name, in registration order.

    Entry-point adapters come first, then the built-in adapters not already
    provided. An entry-point adapter that fails to construct is skipped with a
    warning, and a built-in adapter of that name (if any) takes its place, the
    same outcome :func:`create_adapter` gives for that name.
    """
    adapters: Dict[str, Any] = {}
    for name, factory in entry_point_adapter_factories().items():
        try:
            adapters[name] = factory()
            logger.debug(f"Registered adapter '{name}' from entry point")
        except Exception as e:
            logger.warning(f"Failed to load adapter '{name}' from entry point: {e}")
    for name, factory in builtin_adapter_factories().items():
        if name not in adapters:
            adapters[name] = factory()
    return adapters


def load_with_adapter(name: str, source: Any, **kwargs: Any) -> Any:
    """Load ``source`` through the adapter registered as ``name``.

    Equivalent to what ``DataManagerV2.load_modality`` does with a file path
    (without its optional validation, which only checks the result), so an
    exported notebook starts from the same object the session loaded.
    """
    return create_adapter(name).from_source(source, **kwargs)

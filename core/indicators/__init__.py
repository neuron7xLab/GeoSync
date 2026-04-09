# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Public indicator exports for convenient access in tests and notebooks."""
from __future__ import annotations

from core.utils.optional_dependency import MissingOptionalDependency

from .cache import (
    BackfillState,
    CacheRecord,
    FileSystemIndicatorCache,
    cache_indicator,
    hash_input_data,
    make_fingerprint,
)
from .ensemble_divergence import (
    EnsembleDivergenceResult,
    IndicatorDivergenceSignal,
    compute_ensemble_divergence,
)

try:  # optional chain may pull heavy data stack in slim runtimes
    from .hierarchical_features import (
        FeatureBufferCache,
        HierarchicalFeatureResult,
        TimeFrameSpec,
        compute_hierarchical_features,
    )
except Exception as exc:  # pragma: no cover - optional dependency chain
    FeatureBufferCache = MissingOptionalDependency("FeatureBufferCache", str(exc))  # type: ignore[assignment]
    HierarchicalFeatureResult = MissingOptionalDependency("HierarchicalFeatureResult", str(exc))  # type: ignore[assignment]
    TimeFrameSpec = MissingOptionalDependency("TimeFrameSpec", str(exc))  # type: ignore[assignment]
    compute_hierarchical_features = MissingOptionalDependency("compute_hierarchical_features", str(exc))  # type: ignore[assignment]

from .kuramoto import (
    KuramotoOrderFeature,
    MultiAssetKuramotoFeature,
    compute_phase,
    compute_phase_gpu,
    kuramoto_order,
    multi_asset_kuramoto,
)
from .kuramoto_ricci_composite import (
    GeoSyncCompositeEngine,
    KuramotoRicciComposite,
    MarketPhase,
)
from .multiscale_kuramoto import (
    KuramotoResult,
    MultiScaleKuramoto,
    MultiScaleKuramotoFeature,
    MultiScaleResult,
    TimeFrame,
    WaveletWindowSelector,
)
from .normalization import (
    IndicatorNormalizationConfig,
    IndicatorNormalizer,
    NormalizationMode,
    normalize_indicator_series,
    resolve_indicator_normalizer,
)
from .pipeline import IndicatorPipeline, PipelineResult
from .pivot_detection import (
    DivergenceClass,
    DivergenceKind,
    PivotDivergenceSignal,
    PivotPoint,
    detect_pivot_divergences,
    detect_pivots,
)
from .temporal_ricci import TemporalRicciAnalyzer
from .trading import HurstIndicator, KuramotoIndicator, VPINIndicator

__all__ = [
    "compute_phase",
    "compute_phase_gpu",
    "kuramoto_order",
    "multi_asset_kuramoto",
    "compute_ensemble_divergence",
    "IndicatorDivergenceSignal",
    "EnsembleDivergenceResult",
    "KuramotoOrderFeature",
    "MultiAssetKuramotoFeature",
    "KuramotoIndicator",
    "HurstIndicator",
    "VPINIndicator",
    "FeatureBufferCache",
    "HierarchicalFeatureResult",
    "TimeFrameSpec",
    "compute_hierarchical_features",
    "IndicatorPipeline",
    "PipelineResult",
    "BackfillState",
    "CacheRecord",
    "FileSystemIndicatorCache",
    "cache_indicator",
    "hash_input_data",
    "make_fingerprint",
    "KuramotoResult",
    "MultiScaleKuramoto",
    "MultiScaleKuramotoFeature",
    "MultiScaleResult",
    "TimeFrame",
    "WaveletWindowSelector",
    "KuramotoRicciComposite",
    "MarketPhase",
    "GeoSyncCompositeEngine",
    "TemporalRicciAnalyzer",
    "IndicatorNormalizer",
    "IndicatorNormalizationConfig",
    "NormalizationMode",
    "normalize_indicator_series",
    "resolve_indicator_normalizer",
    "PivotPoint",
    "PivotDivergenceSignal",
    "DivergenceClass",
    "DivergenceKind",
    "detect_pivots",
    "detect_pivot_divergences",
]

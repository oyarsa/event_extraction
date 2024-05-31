from self_critique.metric.fgcr_metric_cls import FGCRCls
from self_critique.metric.maven_metric_cls import Maven
from self_critique.metric.maven_straight_metric_cls import MavenStraight
from self_critique.metric.reconstruct_metric import ReconstructMetric
from self_critique.metric.types import MetricPrediction, MetricReference


def get_metrics(
    mode: str, references: list[MetricReference], predictions: list[MetricPrediction]
) -> dict[str, float]:
    return {
        "fcr": FGCRCls,
        "extract": FGCRCls,
        "maven": Maven,
        "maven_s": MavenStraight,
        "reconstruct": ReconstructMetric,
    }[mode]._compute(references=references, predictions=predictions)


__all__ = [
    "FGCRCls",
    "ReconstructMetric",
    "Maven",
    "MavenStraight",
    "MetricPrediction",
    "MetricReference",
    "get_metrics",
]

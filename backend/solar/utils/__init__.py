from utils.config import Config

def __getattr__(name):
    if name == "SolarTransforms":
        from utils.transforms import SolarTransforms
        return SolarTransforms
    if name == "EDAVisualizer":
        from utils.visualizer import EDAVisualizer
        return EDAVisualizer
    if name == "AugmentationVisualizer":
        from utils.visualizer import AugmentationVisualizer
        return AugmentationVisualizer
    if name == "PredictionVisualizer":
        from utils.visualizer import PredictionVisualizer
        return PredictionVisualizer
    if name == "ComparisonVisualizer":
        from utils.visualizer import ComparisonVisualizer
        return ComparisonVisualizer
    raise AttributeError(f"module 'utils' has no attribute {name!r}")

__all__ = [
    "Config",
    "SolarTransforms",
    "EDAVisualizer",
    "AugmentationVisualizer",
    "PredictionVisualizer",
    "ComparisonVisualizer",
]

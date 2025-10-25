
"""
AutoSplat Utils
===============

Utility components supporting the AutoSplat rendering framework.

This subpackage currently provides:
- **RenderObject**: a high-level Keras layer that manages multiple BlenderLayer
  instances and produces full RGB image outputs from Gaussian splats.

Future additions may include:
- Visualization helpers
- Profiling utilities
- Scene export/import tools

Example
-------
>>> from autosplat.utils import RenderObject
>>> renderer = RenderObject(blend_layers=[layer1, layer2])
>>> img = renderer(blend_id=0)
>>> renderer.preview(save_path="renders/", blend_id=0)
✅ Saved render at renders/render_0.png
"""

from .render import RenderObject

__all__ = [
    "RenderObject",
]

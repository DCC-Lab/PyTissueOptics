import os
import warnings

from .abstract3DViewer import Abstract3DViewer

AVAILABLE_BACKENDS = ("mayavi", "null")


def get3DViewer() -> Abstract3DViewer:
    backend = os.environ.get("PTO_3D_BACKEND", "mayavi").lower()
    if backend == "mayavi":
        try:
            from .mayavi.mayavi3DViewer import Mayavi3DViewer

            return Mayavi3DViewer()
        except Exception as e:
            warnings.warn(
                "Mayavi is not available. Falling back to a null 3D viewer. Fix the following error to use the Mayavi "
                "backend or select another backend by setting the PTO_3D_BACKEND environment variable (available "
                f"backends: {AVAILABLE_BACKENDS}). \n{e}"
            )
            from .null3DViewer import Null3DViewer

            return Null3DViewer()
    elif backend == "null":
        from .null3DViewer import Null3DViewer

        return Null3DViewer()
    else:
        raise ValueError(f"Invalid backend '{backend}'. Available backends: {AVAILABLE_BACKENDS}")

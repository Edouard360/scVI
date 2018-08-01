from .classification.rf import RF
from .classification.svc import SVC
from .classification.scmap import SCMAP
from .clustering import SEURAT, COMBAT

__all__ = ['RF',
           'SVC',
           'SCMAP',
           'SEURAT'
           'COMBAT']

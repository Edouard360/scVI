from .classification.rf import RF
from .classification.svc import SVC
from .classification.scmap import SCMAP
from .clustering.Seurat import SEURAT
from .clustering.Combat import COMBAT

__all__ = ['RF',
           'SVC',
           'SCMAP',
           'SEURAT'
           'COMBAT']

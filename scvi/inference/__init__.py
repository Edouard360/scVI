from .inference import Inference
from .classifier_inference import ClassifierInference
from .variational_inference import (
    VariationalInference,
    SemiSupervisedVariationalInference,
    AlternateSemiSupervisedVariationalInference,
    JointSemiSupervisedVariationalInference
)
from .experimental_inference import adversarial_wrapper, mmd_wrapper
from .experimental_inference import InfoCatInference, VadeInference, GlowInference

__all__ = ['Inference',
           'ClassifierInference',
           'VariationalInference',
           'SemiSupervisedVariationalInference',
           'AlternateSemiSupervisedVariationalInference',
           'JointSemiSupervisedVariationalInference',
           'adversarial_wrapper',
           'mmd_wrapper',
           'JointSemiSupervisedVariationalInference',
           'InfoCatInference',
           'VadeInference',
           'GlowInference']

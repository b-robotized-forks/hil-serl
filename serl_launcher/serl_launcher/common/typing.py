from typing import Any, Callable, Dict, Sequence, Union

import flax
import jax.numpy as jnp
import numpy as np

try:
    # TensorFlow is optional and only referenced for the ``tf.Tensor`` member of
    # the ``Array`` alias below. Importing it eagerly would force every consumer
    # (including inference-only / torch-free environments) to install
    # TensorFlow, so fall back to ``Any`` when it is unavailable.
    import tensorflow as tf

    _TFTensor = tf.Tensor
except ImportError:
    _TFTensor = Any

PRNGKey = Any
Params = flax.core.FrozenDict[str, Any]
Shape = Sequence[int]
Dtype = Any  # this could be a real type?
InfoDict = Dict[str, float]
Array = Union[np.ndarray, jnp.ndarray, _TFTensor]
Data = Union[Array, Dict[str, "Data"]]
Batch = Dict[str, Data]
# A method to be passed into TrainState.__call__
ModuleMethod = Union[str, Callable, None]

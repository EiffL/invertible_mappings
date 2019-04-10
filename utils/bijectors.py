import tensorflow_probability as tfp
import tensorflow as tf
tfb = tfp.bijectors

from .tfops import invertible_1x1_conv

class Conv1x1_3D(tfb.Bijector):
    """ Implementation of a 1x1 convolution
    """

    def __init__(self, validate_args=False, name='conv1x1_3d', dtype=tf.float32):
        """
        Instantiate bijector
        """
        self._graph_parents = []
        self._name = name
        self._validate_args = validate_args

        super(self.__class__, self).__init__(
          forward_min_event_ndims=0,
          is_constant_jacobian=False,
          validate_args=validate_args,
          dtype=dtype,
          name=name)

    def _forward(self, x):
        y, logdet = invertible_1x1_conv(self._name, x)
        return y

    def _inverse(self, y):
        x, logdet = invertible_1x1_conv(self._name, y, reverse=True)
        return x

    def _forward_log_det_jacobian(self, x):
        y, logdet = invertible_1x1_conv(self._name, x)
        return logdet

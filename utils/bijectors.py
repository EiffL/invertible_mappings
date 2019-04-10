import tensorflow_probability as tfp
import tensorflow as tf
tfb = tfp.bijectors

from .tfops import invertible_1x1_conv, actnorm3d

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

class Actnorm3D(tfb.Bijector):
    """ Implementation of an Actnorm
    """

    def __init__(self, batch_variance=False, logscale_factor=3., scale=1., validate_args=False, name='actnorm_3d', dtype=tf.float32):
        self._graph_parents = []
        self._name = name
        self._validate_args = validate_args
        self._batch_variance = batch_variance
        self._logscale_factor = logscale_factor
        self._scale = scale
        super(self.__class__, self).__init__(
          forward_min_event_ndims=0,
          is_constant_jacobian=False,
          validate_args=validate_args,
          dtype=dtype,
          name=name)

    def _forward(self, x):
        y, logdet = actnorm3d(x, scale=self._scale, logscale_factor=self._logscale_factor,
                              batch_variance=self._batch_variance, reverse=False, scope=self._name)
        return y

    def _inverse(self, y):
        x, logdet = actnorm3d(y, scale=self._scale, logscale_factor=self._logscale_factor,
                              batch_variance=self._batch_variance, reverse=True, scope=self._name)
        return x

    def _forward_log_det_jacobian(self, x):
        y, logdet = actnorm3d(x, scale=self._scale, logscale_factor=self._logscale_factor,
                              batch_variance=self._batch_variance, reverse=False, scope=self._name)
        return logdet

#Squeeze operation for invertible downsampling in 3D
#
#
class Squeeze3d(tfb.Reshape):
    """
    Borrowed from https://github.com/openai/glow/blob/master/tfops.py
    """
    def __init__(self,
                 event_shape_in,
                 factor=2,
                 is_constant_jacobian=True,
                 validate_args=False,
                 name=None):

        assert factor >= 1
        name = name or "squeeze"
        self.factor = factor
        event_shape_out = 1*event_shape_in
        event_shape_out[0] //=2
        event_shape_out[1] //=2
        event_shape_out[2] //=2
        event_shape_out[3] *=8
        self.event_shape_out = event_shape_out

        super(Squeeze3d, self).__init__(
            event_shape_out=event_shape_out,
            event_shape_in=event_shape_in,
        validate_args=validate_args,
        name=name)

    def _forward(self, x):
        if self.factor == 1:
            return x
        factor = self.factor

        shape = tf.shape(x)
        height = shape[1]
        width = shape[2]
        length = shape[3]
        n_channels = x.get_shape()[4]

#         print(height, width, length, n_channels )
#         assert height % factor == 0 and width % factor == 0 and length % factor == 0
        x = tf.reshape(x, [-1, height//factor, factor,
                           width//factor, factor, length//factor, factor, n_channels])
        x = tf.transpose(x, [0, 1, 3, 5, 7, 2, 4, 6])
        x = tf.reshape(x, [-1, height//factor, width//factor,
                               length//factor, n_channels*factor**3])
        return x

    def _inverse(self, x):
        if self.factor == 1:
            return x
        factor = self.factor

        shape = tf.shape(x)
        height = shape[1]
        width = shape[2]
        length = shape[3]
        n_channels = int(x.get_shape()[4])

#         print(height, width, length, n_channels )
        assert n_channels >= 8 and n_channels % 8 == 0
        x = tf.reshape(
            x, [-1, height, width, length, int(n_channels/factor**3), factor, factor, factor])
        x = tf.transpose(x, [0, 1, 5, 2, 6, 3, 7, 4])
        x = tf.reshape(x, (-1, height*factor,
                           width*factor, height*factor, int(n_channels/factor**3)))
        return x

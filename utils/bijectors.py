import tensorflow_probability as tfp
import tensorflow as tf
tfb = tfp.bijectors

from .tfops import invertible_1x1_conv, f_net3d

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





class AffineGlow3d(tfb.Bijector):
    def __init__(self, width=4,
               name="affine"):

#         self.name = name
        self.width = width
        super(AffineGlow3d, self).__init__(
            #is_constant_jacobian=False,
            forward_min_event_ndims=1,
            dtype=None,
            validate_args=False,
            name=name)


    def _forward(self, x):

        n_x = int(x.get_shape()[4])
        assert n_x % 2 == 0

        x1 = x[:, :, :, :, :n_x // 2]
        x2 = x[:, :, :, :, n_x // 2:]

        h = f_net3d(self._name + "/f1", x1, width=self.width, n_out=n_x)
        shift = h[:, :, :, :, 0::2]
        # scale = tf.exp(h[:, :, :, 1::2])
        scale = tf.nn.sigmoid(h[:, :, :, :, 1::2] + 2.)
        x2 += shift
        x2 *= scale
        
        z = tf.concat([x1, x2], 4)
        
        return z
        


    def _inverse(self, x):
        n_x = int(x.get_shape()[4])
        assert n_x % 2 == 0

        x1 = x[:, :, :, :, :n_x // 2]
        x2 = x[:, :, :, :, n_x // 2:]

        h = f_net3d(self._name + "/f1", x1, width=self.width, n_out=n_x)
        shift = h[:, :, :, :, 0::2]

        # scale = tf.exp(h[:, :, :, 1::2])
        scale = tf.nn.sigmoid(h[:, :, :, :, 1::2] + 2.)
        x2 /= scale
        x2 -= shift

        z = tf.concat([x1, x2], 4)


        return z
        

    def _forward_log_det_jacobian(self, x):
        
        shape = x.int_shape(z)
        n_x = shape[4]
        assert n_x % 2 == 0

        x1 = x[:, :, :, :, :n_x // 2]
        x2 = x[:, :, :, :, n_x // 2:]

        h = f_net3d(self._name + "/f1", z1, hps.width, n_x)
        shift = h[:, :, :, :, 0::2]
        # scale = tf.exp(h[:, :, :, 1::2])
        scale = tf.nn.sigmoid(h[:, :, :, :, 1::2] + 2.)
        logdet = tf.reduce_sum(tf.log(scale), axis=[1, 2, 3, 4])
        return logdet

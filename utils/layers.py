import tensorflow as tf
import tensorflow.contrib.slim as slim
import tfops
#from tfops import specnormconv3d, actnorm3d

@slim.add_arg_scope
def wide_resnet(inputs, depth, resample=None,
                keep_prob=None,
                activation_fn=tf.nn.relu,
                is_training=True,
                outputs_collections=None, scope=None):
    """
    Wide residual units as advocated in arXiv:1605.07146
    Adapted from slim implementation of residual networks
    Resample can be 'up', 'down', 'none'
    """
    depth_residual = 2 * depth

    depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
    size_in = inputs.get_shape().as_list()[1]

    with tf.variable_scope(scope, 'wide_resnet', [inputs]) as sc:
        with slim.arg_scope([slim.batch_norm, slim.dropout],
                            is_training=is_training):
            with slim.arg_scope([slim.conv3d_transpose, slim.conv3d],
                                normalizer_fn=slim.batch_norm,
                                activation_fn=activation_fn,
                                weights_initializer=slim.initializers.variance_scaling_initializer(),
                                kernel_size=3,
                                stride=1):

                preact = slim.batch_norm(
                    inputs, activation_fn=activation_fn, scope='preact')

                if resample == 'up':
                    output_size = size_in * 2
                    # Apply bilinear upsampling
                    preact = tf.image.resize_bilinear(
                        preact, [output_size, output_size], name='resize')
                elif resample == 'down':
                    output_size = size_in / 2
                    preact = slim.avg_pool2d(preact,
                                             kernel_size=[2, 2], stride=2,
                                             padding='SAME', scope='resize')

                if depth_in != depth:
                    shortcut = slim.conv3d(
                        preact, depth, kernel_size=1, normalizer_fn=None,
                        activation_fn=None, scope='shortcut')
                else:
                    shortcut = preact

                residual = slim.conv3d(preact, depth_residual, scope='res1')

                if keep_prob is not None:
                    residual = slim.dropout(residual, keep_prob=keep_prob)

                residual = slim.conv3d(residual, depth, stride=1, scope='res2',
                                       normalizer_fn=None, activation_fn=None)

                output = shortcut + residual

                return slim.utils.collect_named_outputs(outputs_collections,
                                                        sc.name,
                                                        output)




@slim.add_arg_scope
def wide_resnet_snorm(inputs, depth, 
                kernel_size=3, stride=1,
                keep_prob=None,
                activation_fn=tf.nn.leaky_relu,
                is_training=True,
                outputs_collections=None, scope=None):
    """
    Wide residual units as advocated in arXiv:1605.07146
    Adapted from slim implementation of residual networks
    Impose Lipschitz control on the kernels by dividing them with
    their spectral norm. Batch norm is replaced by actnorm
    """
    depth_residual = 2 * depth

    depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
    size_in = inputs.get_shape().as_list()[1]

    with tf.variable_scope(scope, 'wide_resnet', [inputs]) as sc:
        with slim.arg_scope([tfops.actnorm3d, slim.dropout],
                            is_training=is_training):
            with slim.arg_scope([tfops.specnormconv3d],
                                kernel_size=kernel_size,
                                stride=stride):

                preact = tfops.actnorm3d(inputs, scope='preact')
                if activation_fn is not None: preact = activation_fn(preact)
                
                if depth_in != depth:
                    shortcut = tfops.specnormconv3d(preact, depth, name='shortcut')
                else:
                    shortcut = preact

                residual = tfops.specnormconv3d(preact, depth_residual, name='res1')
                #residual = specnormconv3d(preact, depth_residual, scope='res1')

                if keep_prob is not None:
                    residual = slim.dropout(residual, keep_prob=keep_prob)

                residual = tfops.specnormconv3d(residual, depth, name='res2')
#                 residual = specnormconv3d(residual, depth, stride=1, scope='res2',
#                                        normalizer_fn=None, activation_fn=None)

                output = shortcut + residual

                return slim.utils.collect_named_outputs(outputs_collections,
                                                        sc.name,
                                                        output)
            





class SpecDenseLayer(tf.keras.layers.Layer):
    '''Return a dense layer with spectral normed weights
    '''
    def __init__(self, num_outputs, activation=None):
        super(SpecDenseLayer, self).__init__()
        self.num_outputs = num_outputs
        if activation is None: activation = tf.identity
        self.activation = activation
#         print(self.name)
    
    def build(self, input_shape):
        with tf.variable_scope('/kernelspecnorm') as scope:
            self.kernel = self.add_variable("kernel", 
                                    shape=[int(input_shape[-1]), 
                                           self.num_outputs])
            self.bias = self.add_variable("bias", 
                                    shape=[self.num_outputs])
    
#     def call(self, input):
#         x = tf.matmul(input, self.kernel)
#         x += self.bias
#         x = self.activation(x)
#         return x

    def call(self, input):
        with tf.variable_scope(self.name+'/kernelspecnorm') as scope:
            if tfops.scope_has_variables(scope):
                scope.reuse_variables()
            x = tf.matmul(input, tfops.spectral_normed_weight(self.kernel, num_iters=3))
            x += self.bias
            x = self.activation(x)
            return x


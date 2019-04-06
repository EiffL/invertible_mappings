import numpy as np
import sys, os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
sys.path.append('../utils/flowpm/')

import tensorflow as tf
from background import *
import tfpm 
import tfpmfuncs as tfpf
from tfpmconfig import Config


pkfile = '../utils/flowpm/Planck15_a1p00.txt'

tf.reset_default_graph()

def linfieldlocal(white, config,  name='linfield'):
    '''generate a linear field with a given linear power spectrum'''

    bs, nc = config['boxsize'], config['nc']
    whitec = tfpf.r2c3d(white, norm=nc**3)
    lineark = tf.multiply(whitec, (pkmesh/bs**3)**0.5)
    linear = tfpf.c2r3d(lineark, norm=nc**3, name=name)
    return linear

#

config = Config(bs=100, nc=32, seed=200, pkfile=pkfile)
bs, nc = config['boxsize'], config['nc']
kmesh = sum(kk**2 for kk in config['kvec'])**0.5
pkmesh = config['ipklin'](kmesh)
print(bs, nc)

xx = tf.placeholder(tf.float32, (nc, nc, nc), name='white')
whitec = tfpf.r2c3d(xx, norm=nc**3)
lineark = tf.multiply(whitec, (pkmesh/bs**3)**0.5)
linear = tfpf.c2r3d(lineark, norm=nc**3, name='linear')
icstate = tfpm.lptinit(linear, config, name='icstate')
fnstate = tfpm.nbody(icstate, config, verbose=False, name='fnstate')
final = tf.zeros_like(linear)
final = tfpf.cic_paint(final, fnstate[0], boxsize=config['boxsize'], name='final')


def relu(x):
    mask =  x<0
    y = x.copy()
    y[mask] *=0
    return y


with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    for i in range(10):
        seed = i
        np.random.seed(seed)
        white = np.random.normal(loc=0, scale=nc**1.5, size=nc**3).reshape(nc, nc, nc) 
        linmesh, finmesh, fstate,istate = sess.run([linear, final, fnstate, icstate], feed_dict={xx:white})
        np.random.seed(seed)
        ovd = finmesh.copy()
        ovd -= 1
        rate = relu(ovd) + 1e-3
        rate[rate>1] = 1
        sample = np.random.poisson(rate)
        try: os.makedirs('../data/toy/L%04d_N%04d/S%04d'%(bs, nc, seed))
        except: pass
        np.save('../data/toy/L%04d_N%04d/S%04d/s'%(bs, nc, seed), linmesh)
        np.save('../data/toy/L%04d_N%04d/S%04d/d'%(bs, nc, seed), finmesh)
        np.save('../data/toy/L%04d_N%04d/S%04d/p'%(bs, nc, seed), sample)
        if i %100==0: print(i)

##print(fstate[0])
##print(linmesh)
##print(finmesh)
##

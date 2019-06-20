import tensorflow as tf
import numpy as np
from scipy import misc
import os
from model import net

inp = tf.placeholder(tf.float32, [None, 224, None, 3])
ar = tf.placeholder(tf.float32, [])
m = net()
res = m.test(inp, ar)

sess = tf.Session()
saver = tf.train.Saver()
saver.restore(sess, 'model_ckpt/model.ckpt')

input_dir = 'input_images'
output_dir = 'results'
r = 0.5		# target aspect ratio

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
img_names = [f for f in os.listdir(input_dir) if f.endswith('.png')]
for img_name in img_names:
    print(img_name)
    img = misc.imread(os.path.join(input_dir, img_name))
    h, w, _ = img.shape
    scale = 224.0 / h
    h = 224
    w = int(w * scale)
    img = misc.imresize(img, [h, w], 'bicubic')
    p_name = os.path.splitext(img_name)[0]
    misc.imsave(os.path.join(output_dir, img_name), img)
    
    tar_w = int(w * r)
    resize_img = misc.imresize(img, [224, tar_w], 'bicubic')
    misc.imsave(os.path.join(output_dir, p_name + '_bic.png'), resize_img)

    img = np.expand_dims(img, 0).astype(np.float32)
    out = sess.run(res, feed_dict={inp: img, ar:r})
    out = np.round(np.clip(out[0], 0, 255)).astype(np.uint8)
    misc.imsave(os.path.join(output_dir, p_name + '_%s.png' % str(r)), out)

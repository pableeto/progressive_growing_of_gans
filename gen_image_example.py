#fake_images_out = G.get_output_for(latents, labels, is_training=True)

import pickle
import numpy as np
import tensorflow as tf


tf.InteractiveSession()

with open('model.pkl', 'rb') as f:
    G, D, Gs = pickle.load(f)

latents = tf.random_normal([1] + Gs.input_shapes[0][1:])
labels = tf.zeros([1] + Gs.input_shapes[1][1:])

fake_data_out = Gs.get_output_for(latents, labels, is_training=False)
fake_data_out = tf.transpose(fake_data_out, [0, 2, 3, 1])
fake_data_tex = 0.5 * (fake_data_out[0,..., 0:3] + 1.0)         #RGB
fake_data_roughness = 0.5 * (fake_data_out[0,..., 3:4] + 1.0)   #Gray
fake_data_normal = fake_data_out[0,..., 4:7]                    #XYZ




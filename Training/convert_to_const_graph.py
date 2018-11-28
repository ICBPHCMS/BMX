import argparse
import os
import sys
import tensorflow as tf
import keras
import numpy
from keras import backend as K

from network import Network

parser = argparse.ArgumentParser()
parser.add_argument('weightfile', metavar='weightfile', type=str, nargs=1,
                    help='a weight file from keras')
arguments = parser.parse_args()


if not os.path.exists(arguments.weightfile[0]):
    print "Error - weight file '",arguments.weightfile[0],"' does not exists"
    sys.exit(1)
if not arguments.weightfile[0].endswith(".hdf5"):
    print "Error - file '",parser.weightfile[0],"' is not a hdf5 file"
    sys.exit(1)

sess = K.get_session()

tf_input_placeholder = tf.placeholder('float32',shape=(None,200,24),name="features")
keras_input = keras.layers.Input(tensor=tf_input_placeholder)

print "input shape: ",keras_input.shape.as_list()

net = Network()
prediction = net.getDiscriminant(keras_input)
model = keras.models.Model(inputs=keras_input,outputs=prediction)
prediction_node = tf.identity(prediction,name="prediction")

init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess.run(init_op)

def shape(tf_tensor):
    dims = tf_tensor.shape.as_list()
    dims[0] = 1
    return dims

model.load_weights(arguments.weightfile[0])

#test if graph can be executed
feed_dict={
    tf_input_placeholder:numpy.zeros(shape(tf_input_placeholder)),
}

prediction_val = sess.run(
    prediction_node,
    feed_dict=feed_dict
)

const_graph = tf.graph_util.convert_variables_to_constants(
    sess,
    sess.graph.as_graph_def(),
    ["prediction"]
)
tf.train.write_graph(const_graph,"",arguments.weightfile[0].replace("hdf5","pb"),as_text=False)

print "Sucessfully saved graph and weights into '%s'"%arguments.weightfile[0].replace("hdf5","pb")


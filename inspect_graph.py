import tensorflow as tf
import os

def _get_base_path():
    if not os.environ.get('OPENPOSE_MODEL', ''):
        return './models'
    return os.environ.get('OPENPOSE_MODEL')

def get_graph_path(model_name):
    return {
        'cmu_640x480': './models/graph/cmu_640x480/graph_opt.pb',
        'cmuq_640x480': './models/graph/cmu_640x480/graph_q.pb',

        'cmu_640x360': './models/graph/cmu_640x360/graph_opt.pb',
        'cmuq_640x360': './models/graph/cmu_640x360/graph_q.pb',

        'mobilenet_thin_432x368': './models/graph/mobilenet_thin_432x368/graph_opt.pb',
    }[model_name]


if __name__ == '__main__':
    model_name = 'mobilenet_thin_432x368'
    graph_path = get_graph_path(model_name)

    with tf.compat.v1.Session() as sess:
        with tf.io.gfile.GFile(graph_path, 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())

        tf.compat.v1.import_graph_def(graph_def, name='')

        for op in sess.graph.get_operations():
            print(op.name)
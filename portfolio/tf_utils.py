"""
TensorFlow / TensorBoard helper functions
"""

import tensorflow as tf

class TensorBoardLogger(object):
    """
    Helper class for tensorboard functionality
    """

    def __init__(self, use_logger = True, path = '', store_frequency = 1):
        self.path = path
        self.store_frequency = store_frequency
        self.use_logger = use_logger

    def initialise(self):
        if self.use_logger == False:
            return
        # Create tensorboard directory
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        self.merged_summary = tf.summary.merge_all()
        self.options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        self.options.output_partition_graphs = True
        self.options.trace_level = tf.RunOptions.SOFTWARE_TRACE
        self.run_metadata = tf.RunMetadata()

    def set_summary_writer(self, sess):
        if self.use_logger == False:
            return
        self.summary_writer = tf.summary.FileWriter(logdir=self.path, graph=sess.graph)

    def write_summary(self, session, feed_dict, iteration, batch_no):
        if (self.use_logger == False) or (iteration % self.store_frequency != 0):
            return
        # The options flag is needed to obtain profiling information
        summary = session.run(self.merged_summary, feed_dict = feed_dict,
                           options=self.options, run_metadata=self.run_metadata)
        self.summary_writer.add_summary(summary, iteration)
        self.summary_writer.add_run_metadata(self.run_metadata, 'iteration %d batch %d' % (iteration, batch_no))

    def write_histogram(self, weights, biases):
        if self.use_logger == False:
            return
        for i, weight in enumerate(weights):
            tf.summary.histogram("weights_%d" % i, weights[i])
        for i, weight in enumerate(biases):
            tf.summary.histogram("biases_%d" % i, biases[i])

    def write_scalar_summary(self, name, tensor):
        tf.summary.scalar(name, tensor)


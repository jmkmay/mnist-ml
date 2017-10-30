import tensorflow as tf

# Import data set for mnist from the tutorials folder in tensorflow install directory
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import rnn


# Declare the mnist object as the input data from the file
mnist = input_data.read_data_sets("/mp/data/", one_hot=True)

num_epochs = 8
n_classes = 10
batch_size = 128

chunk_size = 28
n_chunks = 28
rnn_size = 128

# Declare a global variable that will hold the input and output data
x = tf.placeholder(tf.float32,[None, n_chunks, chunk_size]) # The image is 28x28, therefor the input has 784 neurons
y = tf.placeholder(tf.float32)

# Define the network function that takes an input data
def recurrent_neural_network(x):

    layer = {'weights':tf.Variable(tf.random_normal([rnn_size,n_classes])),
             'biases':tf.Variable(tf.random_normal([n_classes]))}

    x = tf.transpose(x, [1,0,2])
    x = tf.reshape(x, [-1, chunk_size])
    x = tf.split(x, n_chunks, 0)

    lstm_cell = rnn.BasicLSTMCell(rnn_size)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    output = tf.add(tf.matmul(outputs[-1], layer['weights']), layer['biases'])

    #output = tf.nn.relu(output)
    return output

# Define function that trains the network using data from mnist
def train_neural_network(x):
    # Input x (input data) into the network model to get an output calculation
    prediction = recurrent_neural_network(x)
    # Define a cost function handled by tensorflow softmax_cross_entropy_with_logits (no idea what that is)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
    # Define the optimizer function which uses AdamOptimizer minimizing for cost
    # Optional parameter here:  learning_rate = 0.001 by default
    optimizer = tf.train.AdamOptimizer().minimize(cost)


    # Start a tensorflow Session with parameter initialize_all_variables
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Iterate algorithm for previously defined number of epochs
        for epoch in range(num_epochs):
            # Upon each iteration reset the epoch_loss to zero
            epoch_loss = 0
            # For each example within the batch size,
            for _ in range(int(mnist.train.num_examples/batch_size)):
                # Set variables epoch_x and epoch_y to inputs/outputs for current batch
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)

                epoch_x = epoch_x.reshape((batch_size,n_chunks,chunk_size))

                _, c = sess.run([optimizer, cost], feed_dict = {x: epoch_x, y:epoch_y})
                epoch_loss += c
            print('Epoch', epoch, 'completed out of', num_epochs,'loss:', epoch_loss)

        correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))

        accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))
        print('Accuracy:', accuracy.eval({x:mnist.test.images.reshape((-1, n_chunks, chunk_size)), y:mnist.test.labels}))

train_neural_network(x)

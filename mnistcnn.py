import tensorflow as tf

# Import data set for mnist from the tutorials folder in tensorflow install directory
from tensorflow.examples.tutorials.mnist import input_data

# Declare the mnist object as the input data from the file
mnist = input_data.read_data_sets("/mp/data/", one_hot=True)

n_classes = 10

# Batch size of 128 seems to be the standard? (batch sizes of 2^n)
batch_size = 128

# Declare a global variable that will hold the input and output data
x = tf.placeholder('float',[None, 784]) # The image is 28x28, therefor the input has 784 neurons
y = tf.placeholder('float')

def conv2d(x, weights):
    return tf.nn.conv2d(x, weights, strides=[1,1,1,1], padding = 'SAME')

def maxpool2d(x):
    #                          kernel size      movement of the kernel
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


# Define the network function that takes an input data
def cnn_network_model(x):

    weights = {'w_conv1':tf.Variable(tf.random_normal([5, 5, 1, 32])),
               'w_conv2':tf.Variable(tf.random_normal([5, 5, 32, 64])),
               'w_fc':tf.Variable(tf.random_normal([7*7*64, 1024])),
               'out':tf.Variable(tf.random_normal([1024,n_classes]))}

    biases = {'b_conv1':tf.Variable(tf.random_normal([32])),
               'b_conv2':tf.Variable(tf.random_normal([64])),
               'b_fc':tf.Variable(tf.random_normal([1024])),
               'out':tf.Variable(tf.random_normal([n_classes]))}

    x = tf.reshape(x, shape=[-1 ,28, 28, 1])
    conv1 = conv2d(x,weights['w_conv1'])
    conv1 = maxpool2d(conv1)

    conv2 = conv2d(conv1, weights['w_conv2'])
    conv2 = maxpool2d(conv2)

    fc = tf.reshape(conv2,[-1,7*7*64])
    fc = tf.nn.relu(tf.matmul(fc,weights['w_fc'])+biases['b_fc'])

    output = tf.matmul(fc,weights['out']+biases['out'])
    #output = tf.nn.relu(output)
    return output

# Define function that trains the network using data from mnist
def train_neural_network(x):
    # Input x (input data) into the network model to get an output calculation
    prediction = cnn_network_model(x)
    # Define a cost function handled by tensorflow softmax_cross_entropy_with_logits (no idea what that is)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
    # Define the optimizer function which uses AdamOptimizer minimizing for cost
    # Optional parameter here:  learning_rate = 0.001 by default
    optimizer = tf.train.AdamOptimizer(learning_rate = 0.002).minimize(cost)

    num_epochs = 8

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
                _, c = sess.run([optimizer, cost], feed_dict = {x: epoch_x, y:epoch_y})
                epoch_loss += c
            print('Epoch', epoch, 'completed out of', num_epochs,'loss:', epoch_loss)

        correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))

        accuracy = tf.reduce_mean(tf.cast(correct,'float'))
        print('Accuracy:', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

train_neural_network(x)

import tensorflow as tf

# Import data set for mnist from the tutorials folder in tensorflow install directory
from tensorflow.examples.tutorials.mnist import input_data

# Declare the mnist object as the input data from the file
mnist = input_data.read_data_sets("/mp/data/", one_hot=True)

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
batch_size = 100

# Declare a global variable that will hold the input and output data
x = tf.placeholder('float',[None, 784]) # The image is 28x28, therefor the input has 784 neurons
y = tf.placeholder('float')

# Define the network function that takes an input data
def network_model(data_in):

    hidden_layer_1 = {'weights':tf.Variable(tf.random_normal([784, n_nodes_hl1])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_layer_2 = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_layer_3 = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                      'biases':tf.Variable(tf.random_normal([n_classes]))}

    # Feedforward kernel to calculate layers and output based on input data and weights
    # General formula is: layer(N+1) = layer(N)*weights + biases
    l1 = tf.add(tf.matmul(data_in, hidden_layer_1['weights']), hidden_layer_1['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_layer_2['weights']), hidden_layer_2['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_layer_3['weights']), hidden_layer_3['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.add(tf.matmul(l3, output_layer['weights']), output_layer['biases'])

    #output = tf.nn.relu(output)
    return output

# Define function that trains the network using data from mnist
def train_neural_network(x):
    # Input x (input data) into the network model to get an output calculation
    prediction = network_model(x)
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

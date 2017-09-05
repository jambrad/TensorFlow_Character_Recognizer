'''
A module that encapsulates the autoencoding process.
Autoencoding trains the hidden layers to output the input itself
    Ex. (if [0,1] is the input then [0,1] is the expected outcome)
Accepts: one dataset (x => Input Data)
Returns: a tensor of the last hidden layer 

Author: Jhon Christian D. Ambrad
Date: 09/05/17 
'''

import tensorflow as tf

def train(data):
    # Number of inputs
    n_input = data[0].__len__()
    # Number of nodes of the 1st hidden layer
    n_node_h1 = 50
    # Number of nodes of the 2nd hidden layer
    n_node_h2 = 50
    # Number of node of the temporary output layer of 1st hidden layer
    n_node_temp1 = data[0].__len__()
    # Number of node of the temporary output layer of 2nd hidden layer
    n_node_temp2 = n_node_h1
    # Learning rate
    l_rate = 0.05

    # --- Placeholders ---
    # Placeholder for the input data
    x = tf.placeholder('float',[None,n_input])
    # Placeholder for the 1st temporary output layer
    y1 = tf.placeholder('float', [None, n_input])
    # Placeholder for the 2nd temporary output layer
    y2 = tf.placeholder('float', [None, n_node_h1])

    # --- Defining of Models -----
    # Model of the 1st hidden layer
    # Accepts the data from the parameter (data) as the input
    h1 = {'weights': tf.Variable(tf.random_normal([n_input,n_node_h1])),
          'bias': tf.Variable(tf.random_normal([n_node_h1])) }
    # Model of the 2nd hidden layer
    # Accepts the data from the h1 as the input
    h2 = {'weights': tf.Variable(tf.random_normal([n_node_h1,n_node_h2])),
          'bias': tf.Variable(tf.random_normal([n_node_h2])) }
    # Model of the 1st temporary output layer
    # Accepts the data from the h1 as the input
    temp_out1 = {'weights': tf.Variable(tf.random_normal([n_node_h1,n_input])),
          'bias': tf.Variable(tf.random_normal([n_input])) }
    # Model of the 2nd temporary output layer
    # Accepts the data from the h2 as the input
    temp_out2 = {'weights': tf.Variable(tf.random_normal([n_node_h2,n_node_h1])),
          'bias': tf.Variable(tf.random_normal([n_node_h1])) }
    
    # --- Connecting the Network ---
    # Adds the hidden layer 1 in tensorflow
    h_layer_1 = tf.add(tf.matmul(x, h1['weights']),  h1['bias'])
    h_layer_1 = tf.nn.sigmoid(h_layer_1) # Activation function
    # Connects hidden layer 1 to temporary output 1
    # This layer should be one to be optimized first
    o_layer_1 = tf.add(tf.matmul(h_layer_1, temp_out1['weights']),  temp_out1['bias'])
    o_layer_1 = tf.nn.sigmoid(o_layer_1)

    # Adds the hidden layer 2 in tensorflow
    h_layer_2 = tf.add(tf.matmul(h_layer_1, h2['weights']),  h2['bias'])
    h_layer_2 = tf.nn.sigmoid(h_layer_2) # Activation function
    # Connects hidden layer 2 to temporary output 2
    # This layer should be one to be optimized second and be returned
    o_layer_2 = tf.add(tf.matmul(h_layer_2, temp_out2['weights']),  temp_out2['bias'])
    o_layer_2 = tf.nn.sigmoid(o_layer_2)

    # --- Training ---

    # Training hidden layer 1
    prediction_1 = o_layer_1
    # Calculates how far is the prediction to the expected output
    cost_1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = prediction, labels = y))
    # Uses gradient descent to minimize the cost (Updating weights)
    trainer_1 = tf.train.GradientDescentOptimizer(l_rate).minimize(cost)



    


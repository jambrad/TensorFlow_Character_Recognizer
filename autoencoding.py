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
    # Placeholder for the input of hidden layer 2 data
    x2 = tf.placeholder('float',[None,n_node_h1])
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
    o_layer_2 = tf.add(tf.matmul(x2, temp_out2['weights']),  temp_out2['bias'])
    o_layer_2 = tf.nn.sigmoid(o_layer_2)

    # --- Training ---

    # Max iteration of training
    max_epoch = 2000
    # ---Training hidden layer 1---
    prediction_1 = o_layer_1
    # Calculates how far is the prediction to the expected output
    cost_1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = prediction_1, labels = y1))
    # Uses gradient descent to minimize the cost (Updating weights)
    trainer_1 = tf.train.GradientDescentOptimizer(l_rate).minimize(cost_1)

    # Accuracy model for hidden layer 1
    correct_pred_1 = tf.equal(tf.argmax(prediction_1, 1), tf.argmax(y1, 1))
    accuracy_1 = tf.reduce_mean(tf.cast(correct_pred_1, tf.float32))
    # Session for training hidden layer 1
    with tf.Session() as sess:
        # Initializes the global variables
        sess.run(tf.global_variables_initializer())

        for epoch in range(max_epoch):
            
            sess.run(trainer_1, feed_dict={x:data, y1: data})
        print("Hidden layer 1 accuracy : " ,  accuracy_1.eval({x: data, y1: data}))
       
    
    # ---Training hidden layer 2---
    prediction_2 = o_layer_2
    # Calculates how far is the prediction to the expected output
    cost_2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = prediction_2, labels = y2))
    # Uses gradient descent to minimize the cost (Updating weights)
    trainer_2 = tf.train.GradientDescentOptimizer(l_rate).minimize(cost_2)

    # Accuracy model for hidden layer 2
    correct_pred_2 = tf.equal(tf.argmax(prediction_2, 1), tf.argmax(y2, 1))
    accuracy_2 = tf.reduce_mean(tf.cast(correct_pred_2, tf.float32))
    # Session for training hidden layer 1
    with tf.Session() as sess:
        # Initializes the global variables
        sess.run(tf.global_variables_initializer())
        evaluated = h_layer_1.eval({x: data})
        print("Pre-eval: ", evaluated)
        for epoch in range(max_epoch + 1000):
            
            sess.run(trainer_2, feed_dict={x: data, x2:evaluated, y2: evaluated})
        print("Evaluation : ", prediction_2.eval({x: data, x2: evaluated}))
        print("Hidden layer 2 accuracy : " ,  accuracy_2.eval({x: data, x2: evaluated, y2: evaluated}))


train([[1,1],[1,1]])

    


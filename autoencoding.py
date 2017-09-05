'''
A module that encapsulates the autoencoding process.
Autoencoding trains the hidden layers to output the input itself
    Ex. (if [0,1] is the input then [0,1] is the expected outcome)
Accepts: one dataset (x => Input Data)
Returns: a tensor of the last hidden layer 
'''

def train(data):
    
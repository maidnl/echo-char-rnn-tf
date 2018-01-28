# ------------------------------------------------------------------------------
# Project: Understanding RNN
# Author: Daniele Aimo (@maidnl) - maidnl74@gmail.com
# Description: Implementation of a Vanilla RNN in the form tanh(Wx + Us_t-1 + b)
#              The main purpose of this projec is educational so it is written
#              in the hope to be simple and understandable 
# Tested using:
#              Tensorflow version 1.3.0
#              Numpy version 1.13.1
# ------------------------------------------------------------------------------

import sys
import os
import numpy as np
import tensorflow as tf

# NOTE:
# The input data are simply a series of random integers. The output data are the 
# input data delayed ("echoed" after a while)
# So the relation between input and output that the RNN model has to catch is 
# the delay, the sequences of input is important because the output has to 
# replicate the input after a while (a change in the input sequence must be replicated
# in the output sequence)
# 
# input and output are hot encoded

# PARAMETERS to control the script execution
ECHO_DELAY = 3
STATE_DIM = 5
NUM_OF_CLASSES = 3
TRAIN_DATA_DIM = 50000
TEST_DATA_DIM = 50
MAX_TRAIN_SEQUENCE = 15
NUM_OF_EPOCH = 5

# ------------------------------------------------------------------------------
def generate_training_data(num_of_classes = 2, num_of_data = 50000, echo_delay = 3):
    # it generates the training data
    # - num_of_classes is the range of the integers used as input (a num_of_class equal
    #   to 2 means that the random sequences is made up only of 0 and 1, a num_of_class equal
    #   to 3 means that the random sequences is made up 0, 1 and 2, and so on... )
    # - num_of_data is the number of data generated
    # - echo_delay is the delay introduced between the input and the output of the training data

    train_data_x = np.random.randint(num_of_classes,size=num_of_data)
    train_data_y = np.roll(train_data_x,echo_delay)
    train_data_y[0:echo_delay] = 0
    return train_data_x, train_data_y

# ------------------------------------------------------------------------------
def reshape_for_batch(data, batch_dim = 1, batch_len = 15):
    # it calculate the number of possible integer batches of data
    # lenght batch_len
    # for the moment let's ignore batch_dim (here it has no effect but
    # it is introduced because this parameter will be used in other 
    # version of the RNN)
    data_for_each_batch = batch_dim * batch_len 
    num_of_batches = len(data) // data_for_each_batch
    max_data_len = num_of_batches * data_for_each_batch
    data = data[:max_data_len]
    data = data.reshape((batch_dim, -1)) 
    return data, num_of_batches

# ------------------------------------------------------------------------------
class LeftMulModel:
    # class to define a Vanilla RNN using TensorFlow with Left Matrix Multiplication
    # this model is made up by 2 different series of functions
    #
    #  -define_learning_placeholders
    #  -model_learn
    #  -make_learning_graph
    #  -learn
    # are used to define the model for training model
    #
    #  -define_placeholders
    #  -model
    #  -make_model_graph
    #  -execute
    # are used to use the model once it has been successfully trained 
    #
    # The model is the same in both cases (they both share the same define_variables)
    # the learning model need more operations, the comparison between the two might be useful
    
    # ------------------------------------------------------------------------------
    def __init__(self, num_of_classes = 3, train_seq_len = 15, state_size = 5):
        # just init some variables for later use
        
        # the number of epochs for learning
        self.num_epochs = NUM_OF_EPOCH
        # the integer range of the random data sequence used for training (see generate_training_data)
        self.num_of_classes = num_of_classes
        # the maximum lenght of a sequence used during the training of the model
        self.train_seq_len  = train_seq_len
        # the dimensions of the state matrix
        self.state_rows     = state_size
        self.state_cols     = 1

    # ------------------------------------------------------------------------------
    def define_variables(self):
        # define the variables of the model 
        # s_t = tanh(U * x + W * s_t-1 + b)
        self.U    = tf.Variable(np.random.rand(self.state_rows, self.num_of_classes), dtype=tf.float32, name = "input2state")
        self.W    = tf.Variable(np.random.rand(self.state_rows, self.state_rows), dtype=tf.float32, name = "state2state")
        self.bias = tf.Variable(np.zeros((self.state_rows,1)), dtype=tf.float32, name = "bias_state")
        # y = V * s_t + b2
        self.V     = tf.Variable(np.random.rand(self.num_of_classes,self.state_rows),dtype=tf.float32, name = "state2output")
        self.bias2 = tf.Variable(np.zeros((self.num_of_classes,1)), dtype=tf.float32, name = "bias_output")

        self.save_dict = {"U"     : self.U,
                          "W"     : self.W,
                          "bias"  : self.bias,
                          "V"     : self.V,
                          "bias2" : self.bias2 }

        self.saver = tf.train.Saver(self.save_dict)

    # 
    # -------------- FUNCTION FOR LEARNING  
    #

    # ------------------------------------------------------------------------------
    def define_learning_placeholders(self):
        # input
        self.x_placeholder = tf.placeholder(tf.float32, [self.num_of_classes, self.train_seq_len], name="inputs")
        # output (used only during learning)
        self.y_placeholder = tf.placeholder(tf.float32, [self.num_of_classes, self.train_seq_len], name="targets")
        # state
        self.state         = tf.placeholder(tf.float32, [self.state_rows, self.state_cols])

    # ------------------------------------------------------------------------------
    def model_learn(self, inputs_series, output_series):
        # the model "goes forward" for a sequence of input that self.train_seq_len
        # for each step of the sequence is calculated
        # - a new state
        # - an ouput (the logit)
        # - a prediction (softmax of the logit)
        # - a loss (cross entropy between the logit and the desired output)
        # The logits and the predictions are memorize into a list althoug the learning
        # process does not use these lists
        # the training needs the list of the losses because the optimizer will try to
        # minimize the the mean of the all the losses

        self.current_state      = self.state
        self.logits_series      = []
        self.predictions_series = []
        self.loss_series        = []

        for i in range(self.train_seq_len):
            #calculate next state
            next_state = tf.tanh(tf.matmul(self.U,tf.reshape(inputs_series[:,i],(self.num_of_classes,1))) + tf.matmul(self.W, self.current_state) +  self.bias)  
            self.current_state = next_state
            # calculate logits
            logit = tf.matmul(self.V ,self.current_state) + self.bias2
            self.logits_series.append(logit)
            # calculate predictions
            prediction = tf.nn.softmax(logit)
            self.predictions_series.append(prediction)
   
            # just reshape the logits and the label in a form suitable for softmax_cross_entropy_with_logits
            logit = tf.reshape(logit,(1,self.num_of_classes))
            label = output_series[:,i]
            label = tf.reshape(label,(1,self.num_of_classes))

            loss = tf.nn.softmax_cross_entropy_with_logits(logits=logit,labels=label)
            # memorize all losses
            self.loss_series.append(loss)
        # the total loss to be minized is the mean of all the losses in the series
        self.totals_loss = tf.reduce_mean(self.loss_series)
        self.train = tf.train.AdagradOptimizer(0.3).minimize(self.totals_loss)
    
    # ------------------------------------------------------------------------------
    def make_learning_graph(self):
        # make the learning graph
        self.learnin_graph = tf.Graph()

        with self.learnin_graph.as_default():
            self.define_learning_placeholders()
            self.define_variables()
            self.model_learn(self.x_placeholder, self.y_placeholder)

    # ------------------------------------------------------------------------------
    def learn(self, inputs, outputs, num_batches):
        # performs the learning of the model 
        print("...... Start Learning ....... ")
        
        with tf.Session(graph=self.learnin_graph) as sess:
            tf.global_variables_initializer().run()
            min_loss = 1000;
            first_run = True
            for epoch_id in range(self.num_epochs):

                print("Starting new EPOCH", epoch_id)

                if(first_run):
                    # state is initialized to zeros just the first time
                    # otherwise the state calculate by the model itself is used
                    current_state = np.zeros((self.state_rows, self.state_cols))
                    first_run = False

                # go for all possible batches 
                for batch_id in range(num_batches):
                    start_id   = batch_id * self.train_seq_len
                    end_id     = start_id + self.train_seq_len

                    x_batch = inputs[:,start_id:end_id]
                    y_batch = outputs[:,start_id:end_id]
                    # HOT ENCODE input
                    x_bath_hot_enc = np.zeros((self.num_of_classes, self.train_seq_len))
                    x_bath_hot_enc[x_batch,np.arange(self.train_seq_len)] = 1
                    # HOT ENCODE output
                    y_bath_hot_enc = np.zeros((self.num_of_classes, self.train_seq_len))
                    y_bath_hot_enc[y_batch,np.arange(self.train_seq_len)] = 1
                    # run the model
                    total_loss, train, current_state, predictions_series, logits, labels, loss = sess.run(
                    [self.totals_loss, 
                     self.train, 
                     self.current_state, 
                     self.predictions_series, 
                     self.logits_series, 
                     self.predictions_series, 
                     self.loss_series],
                    
                    feed_dict={
                     self.x_placeholder : x_bath_hot_enc,
                     self.y_placeholder : y_bath_hot_enc,
                     self.state         : current_state
                    })
                    # save the model when the total loss decrease
                    if(total_loss < min_loss):
                        min_loss = total_loss
                        save_path = self.saver.save(sess, "./models/LeftMulModel.ckpt")

                    # sometimes print how things are going
                    if batch_id%100 == 0:
                        print("Step ",batch_id, "Loss ", total_loss)             
    
    # 
    # -------------- FUNCTION FOR USE THE MODEL ONCE LEARNED  
    #

    # ------------------------------------------------------------------------------
    def define_placeholders(self):
        # no need for y
        self.x_placeholder = tf.placeholder(tf.float32, [self.num_of_classes, 1], name="inputs")    
        self.state         = tf.placeholder(tf.float32, [self.state_rows, self.state_cols])

    # ------------------------------------------------------------------------------
    def model(self, inp):
        # just the model without the loss calculation, just one step of the sequence each time
        # it assumed that inp is already hot_ecoded on the column
        self.current_state = self.state 
        next_state = tf.tanh(tf.matmul(self.U,inp) + tf.matmul(self.W, self.current_state) +  self.bias)  
        self.current_state = next_state
        self.logit = tf.matmul(self.V ,self.current_state) + self.bias2
        self.prediction = tf.nn.softmax(self.logit,dim=0)

    # ------------------------------------------------------------------------------
    def make_model_graph(self):
        # make the model graph
        self.model_graph = tf.Graph()

        with self.model_graph.as_default():
            self.define_placeholders()
            self.define_variables()
            self.model(self.x_placeholder)

    # ------------------------------------------------------------------------------
    def execute(self,inp):            
        # execute the model for all the sequence contained in inp
        # return the prediction as list 
        with tf.Session(graph=self.model_graph) as sess:
            predictions = []
            # check if the model exist
            if os.path.exists("./models/LeftMulModel.ckpt.index") and os.path.exists("./models/LeftMulModel.ckpt.meta"): 
                # restore the variable learned
                self.saver.restore(sess, "./models/LeftMulModel.ckpt")
                print("Model restored from file.")
            else:
                print("\n\nModel is not available or it has not been trained yet!")
                return None
            # init the state for the first time
            current_state = np.zeros((self.state_rows, self.state_cols))
            # for all inputs execute the model
            for i in range(len(inp)):
                # hot encoding of input
                inp_hot_enc = np.zeros((self.num_of_classes, 1))
                inp_hot_enc[inp[i],0] = 1
                # execute the model a new state and an output are calculated
                _current_state, logit, p = sess.run(
                        [self.current_state, 
                         self.logit, 
                         self.prediction],
                        feed_dict={
                         self.x_placeholder : inp_hot_enc,
                         self.state         : current_state
                        })
                # save all the predictions into a list
                predictions.append(np.argmax(p))
                # copy the new state for the next state
                current_state = _current_state
            
            return predictions  

# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------
                
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    if(len(sys.argv) != 2):
        # check arguments
        print("[ERROR]: Wrong argument number")
        print("This script must be called with 1 argument")
        print("The argument can be:")
        print("    LEARN - to make the RNN learning")
        print("    EXEC  - to see how the model work once learned")
    else:
        model = LeftMulModel(num_of_classes = NUM_OF_CLASSES, train_seq_len = MAX_TRAIN_SEQUENCE, state_size = STATE_DIM)
        # LEARN
        if(sys.argv[1] == "LEARN"):
            # check if the directory for the model exists
            if not os.path.exists("./models"):
                os.mkdir("./models")

            x,y           = generate_training_data(num_of_classes = model.num_of_classes, num_of_data = TRAIN_DATA_DIM, echo_delay = ECHO_DELAY)
            x,_           = reshape_for_batch(x, batch_dim = 1, batch_len = model.train_seq_len)
            y,num_batches = reshape_for_batch(y, batch_dim = 1, batch_len = model.train_seq_len)

            model.make_learning_graph()
            model.learn(x, y, num_batches)
        
        # execute the model 
        elif(sys.argv[1] == "EXEC"):
            # data are generate using the function to generate training (but y are not used)
            x,_  = generate_training_data(num_of_classes = model.num_of_classes, num_of_data = TEST_DATA_DIM, echo_delay = ECHO_DELAY)
            model.make_model_graph()   
            predictions = model.execute(x)
                
            if predictions is not None:
                for i in range(TEST_DATA_DIM):
                    if(i < ECHO_DELAY):
                        print("input = ", x[i] , "output = ", predictions[i] )              
                    else:
                        print("input = ", x[i] , "output = ", predictions[i], " (" , x[i-ECHO_DELAY] , " was the input 3 steps ago)")
        else:
            print("[ERROR]: Wrong argument name")
            print("This script must be called with 1 argument")
            print("The argument can be:")
            print("    LEARN - to make the RNN learning")
            print("    EXEC  - to see how the model work once learned")

   


   
    







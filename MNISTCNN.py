import matplotlib.pyplot as plt
import matplotlib.image as im
import tensorflow as tf
import numpy as np
from sklearn.metrics  import  confusion_matrix
import time
from datetime import timedelta 
import math
from numpy import array
import cv2


#PART 1- LOAD DATA
from mnist import MNIST
data = MNIST(data_dir  = "data/MNIST")


print("Size of:")
print("- Training-set:\t\t{}".format(data.num_train))
print("- Validation-set:\t{}".format(data.num_val))
print("- Test-set:\t\t{}".format(data.num_test))


#CNN layer 1 
filter_size1 = 5
num_filters1 = 16
img_size = data.img_size
num_channels = data.num_channels 
#CNN layer 2 
filter_size2 = 5
num_filters2 = 36

#FC layer
fc_size = 128
num_classes = data.num_classes

#for image_placeholder
img_size_flat = data.img_size_flat
##EXTRA
img_shape = data.img_shape

def new_weigths(shape):
    return tf.Variable(tf.truncated_normal(shape , stddev = 0.05))


def new_biases(length):
    return tf.Variable(tf.constant(0.05 , shape= [length]))

def new_conv_layer(input , num_inputs_channels , filter_size , num_filters,use_pooling = True):
    #make the shape of the weigth
    shapex = [filter_size , filter_size , num_inputs_channels , num_filters]
    weigths = new_weigths(shape = shapex)
    
    #make a bias
    biases = new_biases(length = num_filters)
    
    #create a convultional layer object
    layer = tf.nn.conv2d(input = input , filter = weigths , padding = 'SAME' , strides = [1,1,1,1])
    
    layer += biases
    
    if use_pooling:
        layer = tf.nn.max_pool(value = layer , padding = 'SAME', ksize = [1,2,2,1] , strides = [1,2,2,1])
        layer = tf.nn.relu(layer)
        
    return layer , weigths



def flatten_layer(layer):
    #layer is tf object and have get_shape prperty
    layer_shape = layer.get_shape()
    #layer_shape = [a,b,c,d]
    num_features = layer_shape[1:4].num_elements()
    layer_flat = tf.reshape(layer , [-1 , num_features])
    
    return layer_flat , num_features

def new_fc_layer(input , num_inputs , num_outputs , use_relu = True):
    #make shape of fc weigth
    shapex = [num_inputs , num_outputs]
    weigths = new_weigths(shape = shapex)
    biases = new_biases(length = num_outputs)
    layer = tf.matmul(input , weigths) + biases
    
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer


x = tf.placeholder(tf.float32  , shape =[None , img_size_flat] , name = 'x')
x_image  = tf.reshape(x , shape = [-1,img_size ,img_size , num_channels])

y_true = tf.placeholder(tf.float32 , shape = [None , num_classes] , name = "y_true")
y_true_cls = tf.argmax(y_true , axis = 1)

#pass our images through 1st conv layer
layer_conv1 , weigths_conv1  = new_conv_layer(input = x_image , 
                                              num_inputs_channels = num_channels , 
                                              filter_size = filter_size1,
                                              num_filters = num_filters1)
#pass our images through 2nd conv layer
layer_conv2 , weigths_conv2 =  new_conv_layer(input = layer_conv1 , 
                                              num_inputs_channels = num_filters1 ,
                                              filter_size = filter_size2 ,
                                              num_filters = num_filters2)

layer_flat , num_features= flatten_layer(layer_conv2)

layer_fc1 = new_fc_layer(input = layer_flat ,
                         num_inputs = num_features   ,
                         num_outputs = fc_size , 
                         use_relu = True)


layer_fc2 = new_fc_layer(input = layer_fc1 , 
                         num_inputs = fc_size , 
                         num_outputs = num_classes, 
                         use_relu = False)

y_pred = tf.nn.softmax(layer_fc2)

y_pred_cls = tf.argmax(y_pred , axis = 1)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits= layer_fc2 , 
                                               labels = y_true)
cost = tf.reduce_mean(cross_entropy)

optimizer = tf.train.AdamOptimizer(learning_rate = 1e-4).minimize(cost)


correct_prediction = tf.equal(y_pred_cls , y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction , tf.float32))



#Creating Session
session = tf.Session()

session.run(tf.global_variables_initializer())

# Counter for total number of iterations performed so far.
train_batch_size = 64
total_iterations = 0
def optimize(num_iterations):
    # Ensure we update the global variable rather than a local copy.
    global total_iterations

    # Start-time used for printing time-usage below.
    start_time = time.time()

    for i in range(total_iterations,
                   total_iterations + num_iterations):

        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
        x_batch, y_true_batch, _ = data.random_batch(batch_size=train_batch_size)

        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        session.run(optimizer, feed_dict=feed_dict_train)

        # Print status every 100 iterations.
        if i % 100 == 0:
            # Calculate the accuracy on the training-set.
            acc = session.run(accuracy, feed_dict=feed_dict_train)

            # Message for printing.
            msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"

            # Print it.
            print(msg.format(i + 1, acc))

    # Update the total number of iterations performed.
    total_iterations += num_iterations

    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))





test_batch_size = 1
def print_test_accuracy(show_example_errors=False,
                        show_confusion_matrix=False):

    # Number of images in the test-set.
    num_test = data.num_test

    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.
    cls_pred = np.zeros(shape=num_test, dtype=np.int)

    # Now calculate the predicted classes for the batches.
    # We will just iterate through all the batches.
    # There might be a more clever and Pythonic way of doing this.

    # The starting index for the next batch is denoted i.
    i = 0

    while i < num_test:
        # The ending index for the next batch is denoted j.
        j = min(i + test_batch_size, num_test)

        # Get the images from the test-set between index i and j.
        imagePaths = ['/Users/sandeep/Desktop/Photo on 18-9-2018 at 6.11 PM.jpg']
        datax = np.array([np.array(cv2.imread(imagePaths[0])) for i in range(len(imagePaths))])
        images = datax.flatten().reshape(1, 784)

        # Get the associated labels.
        labels = [0., 0., 0., 0., 0., 0., 0., 0., 1., 0.]

        # Create a feed-dict with these images and labels.
        feed_dict = {x: images,
                     y_true: labels}

        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    # Convenience variable for the true class-numbers of the test-set.
    cls_true = data.y_test_cls

    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)

    # Calculate the number of correctly classified images.
    # When summing a boolean array, False means 0 and True means 1.
    correct_sum = correct.sum()

    # Classification accuracy is the number of correctly classified
    # images divided by the total number of images in the test-set.
    acc = float(correct_sum) / num_test

    # Print the accuracy.
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))

    
optimize(num_iterations=1)
print_test_accuracy()     
        

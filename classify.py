from random import shuffle, random
import numpy as np
import tensorflow as tf
from PIL import Image
import os, shutil
from scipy import ndimage
import sys
import time
import re

img_size = 64

def create_resized_images(input_folder, output_folder, select_files=None, out_xy=(img_size, img_size)):

    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)
    else:
        return

    if not select_files:
        files = os.listdir(input_folder)
    elif isinstance(select_files, list):
        files = select_files
    else:
        raise TypeError("Bad type for input_folder")

    num_files = len(files)

    for infile in files:
        # filename, ext = os.path.basename(infile).split('.')
        filename, ext = os.path.splitext(infile)
        if ext.lower() in [".jpg", ".jpeg"]:

            try:
                im = Image.open(os.path.join(input_folder, infile))
                resized_im = im.resize(out_xy)

                if resized_im.mode != "RGB":  # "RGB" is a 8-bit 3 layered format
                    resized_im.convert("RGB")
                # print(filename)
                resized_im.save(os.path.join(output_folder, filename + ".jpg"))
            except IOError:
                pass
        else:
            print(str(ext.lower()))
    print("Created {0} number of files in {1}".format(num_files, output_folder))


def load():
    '''
    load the images file names from the system and
    divide them to train , validation and testing dataset.
    Return:
    train_list: list of training dataset image names.
    valid_list: list of validation dataset image names.
    test_list: list of testing dataset image names.
    '''

    print('loading dataset')
    train_dir = 'train'
    test_dir = 'test'
    training_files = os.listdir(train_dir)

    shuffle(training_files, random)
    num_images = len(training_files)
    # split=[0.7, 0.3]
    # num_train_list = int(np.round(num_images*split[0]))    # number of images in training set

    # train_list = training_files[:num_train_list]
    # valid_list = training_files[num_train_list:]

    test_list = os.listdir(test_dir)

    return training_files, test_list


def get_label(filename):
    '''
    this function get the label from the file name
    the inctext file name will get [0,1]
    the notinctext file name will get [1,0]
    Parameter:
        file_name: the image file name
    Return:
        [0,1]: inctext
        [1,0]: notinctext
    '''
    classes = {'inctext': [0, 1], 'notinctext': [1, 0]}
    for key in classes.keys():
        if key in filename:
            return classes[key]


def prepare_dataset(folder_name, num_images, is_testing_dataset):
    '''
    This function prepare the dataset in format that is understandable
    by the neural network
    the input tensor will have have the following shape:
    num_images, image_size, image_size2, num_channels
    while the labels will have the following shape:
    num_images*1
    '''
    print('reading dataset')
    dataset = np.ndarray(shape=(num_images, image_size, image_size, num_channels), dtype=np.float32)
    labels = np.ndarray(shape=(num_images, 2), dtype=np.int32)
    image_index = 0
    image_counter = image_index  # added

    lst = sorted(os.listdir(folder_name), key = numericalSort)

    for image_name in lst:
        image_file = os.path.join(folder_name, image_name)
        if image_file.endswith('jpg'):
            if image_counter >= num_images:
                raise Exception('More images than expected: %d >= %d' % (
                    image_counter, num_images))
            try:
                image_data = (ndimage.imread(image_file).astype(float) -
                              pixel_depth / 2) / pixel_depth
                print(image_name)
                if image_data.shape != (image_size, image_size, num_channels):
                    raise Exception('Unexpected image shape: %s' % str(image_data.shape))
                # print('image_data.shape: ',image_data.shape)
                dataset[image_counter, :, :] = image_data
                # print('dataset.shape: ', dataset.shape)
                if not is_testing_dataset:
                    labels[image_counter] = get_label(image_name)
                    #             print(image_name, labels[image_index])
                image_counter += 1
            except IOError as e:
                print(('Could not read:', image_file, ':', e, '- it\'s ok, skipping.'))
    # image_counter = image_index
    dataset = dataset[0:image_counter, :, :, :]
    if not is_testing_dataset:
        labels = labels[0:image_counter]
    print('Full dataset tensor:', dataset.shape)
    print(('Mean:', np.mean(dataset)))
    print(('Standard deviation:', np.std(dataset)))
    print(('Labels:', labels.shape))
    return dataset, labels


numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts



train_dir = 'train'
test_dir = 'test'
num_steps = 10000
if sys.argv and len(sys.argv) > 1:
    num_steps = int(sys.argv[1])
# print 'num_steps='+str(num_steps)
num_images = 0
train_list, test_list = load()
training_num_images = len(train_list)
testing_num_images = len(test_list)
print('step 1 : resizing')
print('resize training images to 64*64 and save it in outtrain folder')
create_resized_images(os.getcwd() + '/train', "outtrain", train_list)
print('resize testing images to 64*64 and save it in outtest folder')
create_resized_images(os.getcwd() + '/test', "outtest", test_list)
print('step 2:')

num_classes = 2  # Number of classes in the classifier
image_size = img_size  # Pixel width and height.
num_channels = 3  # RGB
pixel_depth = 255.0  # Number of levels per pixel.
print('format the dataset to the shape that is expected by neural network')
training_dataset, training_labels = prepare_dataset('outtrain', training_num_images, False)
print('reading testing dataset')
testing_dataset, testing_labels = prepare_dataset('outtest', testing_num_images, True)
del testing_labels
# building neural network
# shuffling
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(training_dataset)))
x_shuffled = training_dataset[shuffle_indices]
y_shuffled = training_labels[shuffle_indices]

training_dataset = x_shuffled
training_labels = y_shuffled
# split the data to training and validation dataset


split = [0.90, 0.10]
num_train_list = int(np.round(len(training_dataset) * split[0]))  # number of images in training set

validation_dataset = training_dataset[num_train_list:]
training_dataset = training_dataset[:num_train_list]

validation_labls = training_labels[num_train_list:]
training_labels = training_labels[:num_train_list]

batch_size = 16  # We are gonna process batch_size images per iteration
patch_size = 5  # Filter size is going to be [patch_size patch_size, num_channels, depth] in the first conv layer
depth = 16
num_hidden = 64
keep_prob = 0.75

graph = tf.Graph()

with graph.as_default():
    # Input data.
    tf_train_dataset = tf.placeholder(
        tf.float32, shape=(batch_size, image_size, image_size, num_channels))  # [16*64*64*3]
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_classes))
    tf_valid_dataset = tf.constant(validation_dataset)
    tf_test_dataset = tf.constant(testing_dataset)

    # Variables.
    layer1_weights = tf.Variable(tf.truncated_normal(
        [patch_size, patch_size, num_channels, depth], stddev=0.1))
    # [11, 11, 3, 16] = [11*11*3(363), 16]
    layer1_biases = tf.Variable(tf.zeros([depth]))
    # [16]
    layer2_weights = tf.Variable(tf.truncated_normal(
        [patch_size, patch_size, depth, depth], stddev=0.1))
    # [11, 11, 16, 16]
    layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))
    # [16]
    layer3_weights = tf.Variable(tf.truncated_normal(
        [image_size // 4 * image_size // 4 * depth, num_hidden], stddev=0.1))
    # [784, 64]
    layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
    # [64]
    layer4_weights = tf.Variable(tf.truncated_normal(
        [num_hidden, num_classes], stddev=0.1))
    # [64, 3]
    layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_classes]))

    # Model.
    def model(data):
        conv = tf.nn.conv2d(data, layer1_weights, [1, 2, 2, 1], padding='SAME')  # [1,2,2,1]
        hidden = tf.nn.relu(conv + layer1_biases)
        conv = tf.nn.conv2d(hidden, layer2_weights, [1, 2, 2, 1], padding='SAME')  # [1,2,2,1]
        hidden = tf.nn.relu(conv + layer2_biases)
        print('hidden: ', hidden.get_shape())
        shape = hidden.get_shape().as_list()
        reshape = tf.reshape(hidden, [-1, shape[1] * shape[2] * shape[3]])
        print('reshape: ', reshape.get_shape())
        hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
        h_dropout = tf.nn.dropout(hidden, keep_prob)
        return tf.matmul(h_dropout, layer4_weights) + layer4_biases


    # Training computation.
    logits = model(tf_train_dataset)
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels))

    # Optimizer.
    # optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
    optimizer = tf.train.AdamOptimizer().minimize(loss)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
    test_prediction = tf.nn.softmax(model(tf_test_dataset))

# In[ ]:

num_steps = num_steps + 1  # num_steps batches are trained

# If num_steps*batch_size is more than the number of images in dataset,
# the training will reloop from index 0
def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])


def save_output(path):
    print('saving into ' + path)
    num_list = range(1, len(inctext_testing_prediction) + 1)
    matrix = np.column_stack((num_list, inctext_testing_prediction))
    np.savetxt(path, matrix, fmt='%1.0f,%0.2f', delimiter=',', header='id,label')


with tf.Session(graph=graph) as session:
    saver = tf.train.Saver()
    tf.global_variables_initializer().run()
    if os.path.exists("model.ckpt"):
        print("restore session .......")
        saver.restore(session, "model.ckpt")
    print('Initialized')
    start_time = time.time()
    for step in range(num_steps):
        offset = (step * batch_size) % (training_labels.shape[0] - batch_size)
        batch_data = training_dataset[offset:(offset + batch_size), :, :, :]
        batch_labels = training_labels[offset:(offset + batch_size), :]
        feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}  # , keep_prob:1.0}
        _, l, predictions = session.run(
            [optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (l < 0.001):
            print('Minibatch loss at step %d: %f' % (step, l))
            print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
            print('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(), validation_labls))
            duration = round((time.time() - start_time) / 60, 2)
            print('Elapsed time(mins): ', duration)
            print('break')
            break
        if (step % 50 == 0):
            print('==================================step:' + str(step) + '==========================')
            print('Minibatch loss at step %d: %f' % (step, l))
            keep_prob = 1.0
            print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
            print('Validation accuracy: %.1f%%' % accuracy(
                valid_prediction.eval(), validation_labls))
            duration = round((time.time() - start_time) / 60, 2)
            print('Elapsed time(mins): ', duration)
    keep_prob = 1.0
    print('training has been Done')
    print('prediction')
    # comput the testing output.
    inctext_testing_prediction = test_prediction.eval()[:, 1]

    save_output('result.csv')

    # saving model
    print('save session to model.ckpt')
    saver.save(session, 'model.ckpt')
    # print(test_prediction.eval())
    # print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
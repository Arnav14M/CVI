import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("/tmp/data/",one_hot="True")
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

batch=100


inp=tf.placeholder(tf.float32,shape=[None,784])
op=tf.placeholder(tf.float32,shape=[None,10])

w1=tf.Variable(tf.ones([784, 100]))
b1=tf.Variable(tf.ones([100]))
w2=tf.Variable(tf.ones([100, 10]))
b2=tf.Variable(tf.ones([10]))

w3=tf.Variable(tf.zeros([784, 10]))
b3=tf.Variable(tf.zeros([10]))


'''Hidden layer uses w1,b1,w2,b2
code without hidden layer uses w3,b3'''
l1=tf.nn.softmax(tf.add(tf.matmul(inp,w1),b1))
ans=tf.nn.softmax(tf.add(tf.matmul(l1,w2),b2))
#ans=tf.nn.softmax(tf.add(tf.matmul(inp,w3),b3))


cost = tf.reduce_mean(tf.square(tf.subtract(ans,op)))
#cost = tf.reduce_mean(-tf.reduce_sum(op * tf.log(ans), reduction_indices=[1]))
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=ans, labels=op))



correct_prediction = tf.equal(tf.argmax(ans,1), tf.argmax(op,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#optimize = tf.train.AdamOptimizer().minimize(cost)
optimize = tf.train.GradientDescentOptimizer(1).minimize(cost)

#summarywriter=tf.summary.FileWriter('./graph')
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(10):
        batch_count = int(mnist.train.num_examples / batch)
        print "Epoch: ", epoch,'cost : ',sess.run(cost, feed_dict={inp: mnist.test.images, op: mnist.test.labels})
        for i in range(batch_count):
            batch_x, batch_y = mnist.train.next_batch(batch)
            sess.run([optimize], feed_dict={inp: batch_x, op: batch_y})
        #sess.run([optimize], feed_dict={inp: batch_x, op: batch_y})
    print "Accuracy: ", accuracy.eval(feed_dict={inp: mnist.test.images, op: mnist.test.labels})  #accuracy for testing data
    #print "Accuracy: ", accuracy.eval(feed_dict={inp: mnist.train.images, op: mnist.train.labels})  #accuraxy for training data
    print "Model Execution Complete"


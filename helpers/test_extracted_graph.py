import tensorflow as tf
import os, time
import numpy as np
import argparse

from tensorflow.python.saved_model import tag_constants
from helpers import read_labels, _mkdir

def run_inference_protobuf_graph(images, batch_size, model_dir):
    
    model_outputs = np.zeros((images.shape[0], 10))
    num_batches = images.shape[0] // batch_size
    
    with tf.Session(graph=tf.Graph(), config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        
        tf.saved_model.loader.load(sess, [tag_constants.SERVING], model_dir)
        prediction = tf.get_default_graph().get_tensor_by_name('output:0')
        inputs = tf.get_default_graph().get_tensor_by_name('input:0')

        mults_node = tf.get_default_graph().get_tensor_by_name('TotalMults:0')
        weights_node = tf.get_default_graph().get_tensor_by_name('TotalWeights:0')
        feed_dict = { inputs: images[0:batch_size] }
        mults, weights = sess.run([mults_node, weights_node], feed_dict=feed_dict)
        
        for step in range(num_batches):
            begin = step*batch_size
            end = begin + batch_size
            feed_dict = { inputs: images[begin:end] }
            model_outputs[begin:end] = sess.run(prediction, feed_dict=feed_dict)
            
    inferred_labels = np.argmax(model_outputs, axis=1)
    return inferred_labels, {'mult':mults[0], 'weights':weights[0]}

def create_submission(labels, network_complexity, model_name="resnet"):
    import datetime
    now = time.time()
    now_str = datetime.datetime.fromtimestamp(now).strftime('%m%d-%H%M%S')
    _mkdir('submissions')
    with open('submissions/submission-%s-%d-%d-%s.csv' % (model_name, network_complexity['weights'], network_complexity['mult'], now_str), 'w') as f:
        f.write("Id,Prediction\n")
        for n in range(labels.shape[0]):
            f.write("%d,%d\n" % (n,labels[n]))

def get_score(accuracy, network_complexity):
    aggregate_complexity = np.log10((network_complexity['mult']/50+network_complexity['weights'])/1e5)
    error_rate = (1-accuracy)*100
    loss = (error_rate + 10*aggregate_complexity)
    
    if error_rate > 50:
        return (0, loss)
    else:
        score = 0
        if network_complexity['mult'] < 5e7:
            score += 15
        if network_complexity['weights'] < 1e6:
            score += 15
            
        if error_rate <= 5:
            score += 90
        elif error_rate <= 10:
            score += 80
        elif error_rate <= 20:
            score += 70
        elif error_rate <= 30:
            score += 60
        elif error_rate <= 40:
            score += 50
        else:
            score += 40
            
        return (score, loss)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', help='where to find exported model.')
    parser.add_argument('--model_name', help='name of model to include in submission file')
    parser.add_argument('--use_training_pixel_mean', help='use training pixel mean')
    args = parser.parse_args()

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    train_val_labels = read_labels("dataset/train_labels.csv")
    print("Loading training/validation data...")
    train_val_images = np.load("dataset/train_data.npy")

    all_images_mean = np.mean(train_val_images, axis=0)
    train_val_images = train_val_images - all_images_mean

    train_labels = train_val_labels[:40000]
    val_labels = train_val_labels[40000:]
    all_labels = {'train': train_labels, 'validation': val_labels}

    train_images = train_val_images[:40000]
    val_images = train_val_images[40000:]
    all_images = {'train': train_images, 'validation': val_images}
    batch_size = 128

    print("Running inference on validation...")
    validation_result, network_complexity = run_inference_protobuf_graph(val_images, batch_size, args.model_dir)
    print(set(validation_result), validation_result[0:10], val_labels[0:10])
    validation_accuracy = np.sum(validation_result == val_labels)*1.0/val_labels.shape[0]
    print("Validation Accuracy %g%%" % (validation_accuracy*100), "Network Complexity:", network_complexity)

    base_score, loss = get_score(validation_accuracy, network_complexity)
    print("Base Score (Validation):", base_score)

    print("Loading testing data...")
    test_images = np.load("dataset/test_data.npy")

    use_training_pixel_mean = True if args.use_training_pixel_mean == 'true' else False
    test_images_mean = np.mean(test_images, axis=0)
    if use_training_pixel_mean: 
        print("using training pixel mean")
        test_images = test_images - all_images_mean
    else:
        print("using test pixel mean")
        test_images = test_images - test_images_mean

    print("Running inference on testing...")
    test_result, _ = run_inference_protobuf_graph(test_images, batch_size, args.model_dir)
    create_submission(test_result, network_complexity, args.model_name)

    print("Done!")


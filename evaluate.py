import tensorflow as tf
from model import wide_resnet
from dataset import DataGenerator
import argparse
from scipy.special import softmax
import numpy as np


def load_cifar10():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = (x_train.astype('float32') / 256 ) - 0.5
    x_test =(x_test.astype('float32') / 256 ) - 0.5
    y_train=tf.keras.utils.to_categorical(y_train, num_classes = 10)
    y_test=tf.keras.utils.to_categorical(y_test, num_classes = 10)

    return x_train, y_train, x_test, y_test

def accuracy(pred,labels):
    probs = softmax(pred)
    probs = np.mean(probs,axis=1)
    accuracy = tf.keras.metrics.categorical_accuracy(labels,probs) #Funkar detta?
    accuracy=np.mean(accuracy)
    return accuracy

def nll(pred,labels,ensemble_size):
    tiled_labels = np.tile(np.expand_dims(labels,1),[ensemble_size,10]) #10 or 1? 

    return tf.keras.losses.categorical_crossentropy(tiled_labels,pred,from_logits=True)

def member_accuracy(pred,labels,ensemble_size):
    probs = softmax(pred)
    accuracy = []
    for i in range(ensemble_size):
        member_probs = probs[:,i]
        accuracy.append(tf.keras.metrics.categorical_accuracy(labels,member_probs))
        
    return accuracy

def member_nll(pred,labels,ensemble_size):
    probs = softmax(pred)
    nll = []
    for i in range(ensemble_size):
        member_probs = probs[:,i]
        nll.append(tf.keras.losses.categorical_crossentropy(labels,member_probs))
    return nll

def main(args):
    
    if args.dataset == "cifar-10":
        x_train,y_train, x_test,y_test = load_cifar10()
        n_classes = 10
    

    input_shape = tuple([args.ensemble_size]+ list(x_train[0].shape))

    test_data=DataGenerator(x_test,y_test,args.batch_size,args.batch_rep,args.inp_rep_prob,args.ensemble_size, False, n_classes)

    model = wide_resnet(input_shape,args.d,args.w_mult,n_classes, args.l2_reg)
    
    model.load_weights(args.model)

    y_pred = model.predict(test_data)
    #print(y_pred.shape)
    print(f"Accuracy: {accuracy(y_pred,y_test)}")
    print(f"NLL: {nll(y_pred,y_test,args.ensemble_size)}")
    print(f"Member Accuracy: {member_accuracy(y_pred,y_test,args.ensembel_size)}")
    print(f"Member NLL: {member_nll(y_pred,y_test,args.ensembel_size)} ")

if __name__=="__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--model',type=str, required= True)

    parser.add_argument('--ensemble_size',type=int, default=3, required=False)
    parser.add_argument('--batch_size',type=int,default=512, required= False)
    parser.add_argument('--gpus',type=int,default=4, required=False)
    parser.add_argument('--batch_rep',type=int,default=4, required=False)
    parser.add_argument('--inp_rep_prob',type=float,default=0.5, required= False)
    parser.add_argument('--l2_reg',type=int,default=3e-4, required=False)
    parser.add_argument('--d',type=int,default=28, required=False)
    parser.add_argument('--w_mult',type=int,default=10, required=False)
    parser.add_argument('--dataset',type=str,default="cifar-10", required=False)

    args=parser.parse_args()


    main(args)
import tensorflow as tf
from model import wide_resnet
from dataset import DataGenerator
import argparse

def load_cifar10():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = (x_train.astype('float32') / 256 ) - 0.5
    x_test =(x_test.astype('float32') / 256 ) - 0.5

    return x_train, y_train, x_test, y_test

def accuracy(pred,label):

    return 

def main(args):
    
    if args.dataset == "cifar-10":
        x_train,y_train, x_test,y_test = load_cifar10()
        n_classes = 10

    input_shape = tuple([3]+ list(x_train[0].shape))

    test_data=DataGenerator(x_test,y_test,args.batch_size,args.batch_rep,args.inp_rep_prob,args.ensemble_size,True)

    model = wide_resnet(input_shape,args.d,args.w_mult,n_classes, args.l_2)
    
    model.load_weights(args.model)

    y_pred = model.predict(test_data)

    accuracy(y_pred,y_test)

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


    main()
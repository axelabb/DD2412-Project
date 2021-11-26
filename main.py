import tensorflow as tf
import argparse
import os
from model import wide_resnet
from dataset import DataGenerator

from metrics import NLL

def load_cifar10():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = (x_train.astype('float32') / 256 ) - 0.5
    x_test =(x_test.astype('float32') / 256 ) - 0.5

    return x_train, y_train, x_test, y_test


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('ensemble_size',type=int,default=3)
    parser.add_argument('epochs',type=int,default=250)
    parser.add_argument('batch_size',type=int,default=512)
    parser.add_argument('gpus',type=int,default=4)
    parser.add_argument('batch_rep',type=int,default=4)
    parser.add_argument('inp_rep_prob',type=float,default=0.5)
    parser.add_argument('l2_reg',type=int,default=3e-4)
    parser.add_argument('d',type=int,default=28)
    parser.add_argument('w_mult',type=int,default=10)
    parser.add_argument('dataset',type=str,default="cifar-10")

    args=parser.parse_args()

    if args.dataset == "cifar-10":
        x_train,y_train, x_test,y_test = load_cifar10()
        n_classes = 10


    input_shape = tuple([3]+ list(x_train[0].shape))
    
    steps_per_epoch = x_train.shape[0] //args.batch_size

    traing_data=DataGenerator(x_train,y_train,args.batch_size,args.batch_rep,args.inp_rep_prob,args.ensemble_size,True)

    # Define the checkpoint directory to store the checkpoints.
    checkpoint_dir = './training_checkpoints'
    # Define the name of the checkpoint files.
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
    callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,
                                       save_weights_only=True)]


    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        model = wide_resnet(input_shape,args.d,args.w_mult,n_classes, args.l_2)

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            0.1,
            decay_steps=steps_per_epoch,
            decay_rate=0.1)

        optimizer = tf.keras.optimizers.SGD(lr_schedule)

        model.compile(optimizer,loss = NLL())
        model.fit(traing_data,epochs=args.epochs,,callbacks=callbacks)

    

    

if __name__=="__main__":
    main()
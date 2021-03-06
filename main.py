import tensorflow as tf
import argparse
import os
from model import wide_resnet
from dataset import DataGenerator
from metrics import NLL,NLL_metric
import pickle
from tensorflow.keras import backend as K

def load_cifar10():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = (x_train.astype('float32') / 256 ) - 0.5
    x_test =(x_test.astype('float32') / 256 ) - 0.5
    y_train=tf.keras.utils.to_categorical(y_train, num_classes = 10)
    y_test=tf.keras.utils.to_categorical(y_test, num_classes = 10)

    return x_train, y_train, x_test, y_test


def main(args):


    if args.dataset == "cifar-10":
        x_train,y_train, x_test,y_test = load_cifar10()
        n_classes = 10


    input_shape = tuple([args.ensemble_size]+ list(x_train[0].shape))
    
    steps_per_epoch = x_train.shape[0] //args.batch_size

    traing_data=DataGenerator(x_train,y_train,args.batch_size,args.batch_rep,args.inp_rep_prob,args.ensemble_size,True, n_classes)
    
    if args.noise:
        traing_data.add_label_noise(args.noise,0.4)

    if args.sigma:
        traing_data.noise_againse_noise(args.sigma)

    # Define the checkpoint directory to store the checkpoints.
    checkpoint_dir = './training_checkpoints'
    # Define the name of the checkpoint files.
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
    callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,
                                       save_weights_only=True)]


    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        model = wide_resnet(input_shape,args.d,args.w_mult,n_classes, args.l2_reg)

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            0.1,
            decay_steps=steps_per_epoch,
            decay_rate=0.9)

        optimizer = tf.keras.optimizers.SGD(lr_schedule,momentum=0.9,nesterov=True)
        model.compile(optimizer,loss = NLL())
        history = model.fit(traing_data,epochs=args.epochs,callbacks=callbacks)
        if args.noise:
            sigma = 0
            if args.sigma:
                sigma = args.sigma
            model.save(f"model_{args.noise}_sigma{sigma}.h5")
            with open(f"model_{args.noise}_sigma{sigma}.history", 'wb') as file_pi:
                pickle.dump(history.history, file_pi)
        else:
            model.save(f"model_M{args.ensemble_size}__br{args.batch_rep}_ir{args.inp_rep_prob}.h5")
            with open(f"model_M{args.ensemble_size}__br{args.batch_rep}_ir{args.inp_rep_prob}.history", 'wb') as file_pi:
                pickle.dump(history.history, file_pi)
        #joblib.dump(history, f"model_M{args.ensemble_size}__br{args.batch_rep}_ir{args.inp_rep_prob}.history")


    

if __name__=="__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--ensemble_size',type=int, default=3, required=False)
    parser.add_argument('--epochs',type=int,default=250, required= False)
    parser.add_argument('--batch_size',type=int,default=512, required= False)
    parser.add_argument('--gpus',type=int,default=4, required=False)
    parser.add_argument('--batch_rep',type=int,default=4, required=False)
    parser.add_argument('--inp_rep_prob',type=float,default=0.5, required= False)
    parser.add_argument('--l2_reg',type=int,default=3e-4, required=False)
    parser.add_argument('--d',type=int,default=28, required=False)
    parser.add_argument('--w_mult',type=int,default=10, required=False)
    parser.add_argument('--dataset',type=str,default="cifar-10", required=False)

    parser.add_argument('--sigma', type = float, default=None, required=None)
    parser.add_argument('--noise',type=str,default=None)

    args=parser.parse_args()


    main(args)
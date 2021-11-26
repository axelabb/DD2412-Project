import tensorflow as tf

from model import wide_resnet
from dataset import DataGenerator

from metrics import NLL

def load_cifar10():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = (x_train.astype('float32') / 256 ) - 0.5
    x_test =(x_test.astype('float32') / 256 ) - 0.5

    return x_train, y_train, x_test, y_test


def main():

    x_train,y_train, x_test,y_test = load_cifar10()

    ensemble_size = 3
    d = 28
    w_mult = 10
    n_classes = 10
    epochs = 250
    batch_size = 128
    batch_rep = 4
    inp_rep_prob = 0.5
    input_shape = tuple([3]+ list(x_train[0].shape))
    val_split = 0.1
    l_2 = 3e-4
    steps_per_epoch = x_train.shape[0] * 1 - val_split

    traing_data=DataGenerator(x_train,y_train,batch_size,batch_rep,inp_rep_prob,ensemble_size,True)

    model = wide_resnet(input_shape,d,w_mult,n_classes, l_2)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        0.1,
        decay_steps=steps_per_epoch,
        decay_rate=0.1)

    optimizer = tf.keras.optimizers.SGD(lr_schedule)

    model.compile(optimizer,loss = NLL())
    model.fit(traing_data,epochs=epochs)

    

    

if __name__=="__main__":
    main()
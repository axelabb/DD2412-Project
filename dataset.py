
import numpy as np
import tensorflow as tf

class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self,X,y,batch_size,batch_rep,inp_rep_prob,ensemble_size,training,n_classes,shuffle=True):
        self.X = X
        self.y = y
        if training:
            self.batch_size = batch_size //batch_rep
        else:
            self.batch_size = batch_size
        self.shuffle = shuffle
        self.ensemble_size = ensemble_size
        self.batch_rep = batch_rep
        self.inp_rep_prob = inp_rep_prob
        self.n = X.shape[0]
        self.training = training
        self.n_classes = n_classes
        self.noise = False

    def on_epoch_end(self):
        if self.shuffle:
            idxs = np.arange(self.n)
            np.random.shuffle(idxs)
            self.X = self.X[idxs]
            self.y = self.y[idxs]

    def __get_train_data(self,imgs,labels):

        batch_rep = np.tile(np.arange(imgs.shape[0]),[self.batch_rep])
        np.random.shuffle(batch_rep)
        input_shuffle=int(batch_rep.shape[0] * (1. - self.inp_rep_prob))
        #Kan detta göras bättre?
        shuffle_idxs = [np.concatenate([np.random.permutation( batch_rep[:input_shuffle]), batch_rep[input_shuffle:]]) for _ in range(self.ensemble_size)]
        if self.noise:
            labels += self.sigma*np.random.randn(*labels.shape)
        imgs = np.stack([np.take(imgs, indxs, axis=0) for indxs in shuffle_idxs], axis=1)
        labels = np.stack([np.take(labels, indxs, axis=0) for indxs in shuffle_idxs], axis=1)
        
        return imgs, labels

    def __get_test_data(self,imgs,labels):
        imgs = np.tile(np.expand_dims(imgs, 1), [1, self.ensemble_size, 1, 1, 1])
        
        return imgs, labels

    def __getitem__(self, index):
        to = (index + 1) * self.batch_size
        if to> self.n:
            to = self.n
        imgs = self.X[index * self.batch_size : to]
        labels = self.y[index * self.batch_size: to]

        if self.training:
            imgs, labels = self.__get_train_data(imgs,labels)
        else:
            imgs,labels = self.__get_test_data(imgs,labels)
        return  imgs, labels
        
    def __len__(self):
        return int(np.ceil(self.n / self.batch_size))

    def add_label_noise(self,noise,ratio):
        y = np.argmax(self.y,axis=1)

        for i in range(len(y)):
            if noise=='sym':
                p1 = ratio/(self.n_classes-1)*np.ones(self.n_classes)
                p1[y[i]] = 1-ratio
                y[i] = np.random.choice(self.n_classes,p=p1)

            elif noise=='asym':
                y[i] = np.random.choice([y[i],(y[i]+1)%self.n_classes],p=[1-ratio,ratio])   

        self.y = np.eye(self.n_classes)[y]

    def noise_againse_noise(self,sigma):
        self.noise = True
        self.sigma = sigma
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This approach starts with the embeddings from glove and then
attempts to learn refined embeddings by means of a CNN.

Architecture:
    embedding layer
    -> convolution -> ReLU -> spatial pyramid pooling
    -> fully connected layer with dropout
    -> fully connected layer (readout layer)
    -> softmax

@author: koradir
"""

import tensorflow as tf
import numpy as np
import numpy.random as nprand

try:
   import cPickle as pickle
except:
   import pickle
   
import os

from SpatialPyramidPooling import spatial_pyramid_pool
from PARAMS import CNN_PARAMS as PARAMS
from statusbar import status_update


class CNN_TweetClassifier:
    
    def __init__(self,
                 vocab='vocab.pkl',  # saved dictionary (word -> word number)
                 embeddings=None,
                 save_as='./cnn_classifier.ckpt',  # save to this file
                 saved_model='./cnn_classifier.ckpt',  # set to None to discard saved model
                 debug=False
                 ):
        
        self.debug = debug
        self._save_as = save_as
        
        """LOAD VOCABULARY"""
        self.vocab = {}
        with open(vocab,'rb') as f:
            self.vocab = pickle.load(f)
        print("vocabulary loaded")
        
        """BUILD MODEL"""
        self._model = None
        self._train_step = None
        self._accuracy = None
        self._tf_variables = []
        
        '''closing a session does not get rid of legacy stuff'''
        tf.reset_default_graph()
        
        '''builds graph and initialises above variables'''
        self._build_model(embeddings)
        
        """INITIALISE TF VARIABLES"""
        model_restorable = (
                saved_model is not None
                and os.path.exists(f'{saved_model}.index')
                )
            
        
        if self.debug:
            print('restore from file: {0}'.format('yes' if model_restorable else 'no'))
        
        self._session = tf.Session()
        
        saver = tf.train.Saver(self._tf_variables)
        with self._session.as_default():
            if model_restorable:
                tf.global_variables_initializer().run()
                saver.restore(self._session,saved_model)
            else:
                tf.global_variables_initializer().run()
                #saver.save(self._session,save_as,write_meta_graph=False)
                
        if self.debug:
            assert not self._session._closed
    
    def __del__(self):
        print("CALLING __del__")
        if not self._session._closed:
            self._session.close()
        
    def _build_model(self,fixed_embeddings=None):
        '''Like in Kim's "Convolutional Neural Network for Sentence Classification",
           (http://www.aclweb.org/anthology/D14-1181),
           we use filters of different sizes to capture different 'n-grams'.
           
           Kim's approach is to extract from each filter a single feature
           by means of a max-over-time pooling operation.
           
           The reasoning behind this move is that "this pooling scheme naturally
           deals with variable sentence lengths"
           
           However, about a year after Kim's paper, Kaiming et al published their
           vision of what they call "Spatial Pyramid Pooling (SPP)",
           (https://arxiv.org/pdf/1406.4729.pdf),
           with the express intent on avoiding padding or otherwise distorting the
           samples.
           
           SPP yields a fixed-size output, independent of the input size, and
           therefore provides a different way to deal with variable sentence lengths - 
           one that does not require limiting ourselves to a single feature
           per filter.
        '''
        def conv2d(x,W,stride_x=1,stride_y=PARAMS.dim_embeddings):
            '''
            2D convolution, expects 4D input x and filter matrix W
            Here, stride equals one word.
            '''
            return tf.nn.conv2d(x,W,strides=[1,stride_x,stride_y,1],padding ='SAME')
        
        def weight_variable(shape,name):
            '''
            Initialise weights with a small amount of noise for symmetry breaking
            '''
            initial = tf.truncated_normal(shape,stddev=0.1)
            v = tf.Variable(initial,name=name)
            self._tf_variables.append(v)
            return v
        
        def bias_variable(shape,name):
            '''
            Initialise biases slightly positive to avoid "dead" neurons.
            '''
            initial = tf.constant(0.1,shape=shape)
            v = tf.Variable(initial,name=name)
            self._tf_variables.append(v)
            return v
        
        def EmbeddingVariable(shape,name):
            initial = tf.truncated_normal(shape)
            v = tf.Variable(initial,name=name)
            self._tf_variables.append(v)
            return v
        
        self._x_input = tf.placeholder(tf.int32, shape=[PARAMS.batch_size, None])  # lists of tokens
        self._y_input = tf.placeholder(tf.int32, shape=[PARAMS.batch_size, PARAMS.nof_classes])  # expect one-hot
        
        """EMBEDDING"""
        if fixed_embeddings is None:
            embeddings = EmbeddingVariable([len(self.vocab), PARAMS.dim_embeddings],'embeddings')
        else:
            E = np.load(fixed_embeddings)
            assert E.shape == (len(self.vocab), PARAMS.dim_embeddings), f"expected embeddings matrix of size {len(self.vocab), PARAMS.dim_embeddings} but got {E.shape}"
            embeddings = tf.constant(E,dtype=tf.float32)
        
        h_lookup = tf.nn.embedding_lookup(embeddings,self._x_input)
        # note: h_embed has dimensions batch_size x sentence_length x dim_embeddings
        
        if self.debug:
            print('h_lookup:',h_lookup.get_shape())
            
        embedding_weights = weight_variable([len(self.vocab),1],'embedding_weights')
        h_weights = tf.nn.embedding_lookup(embedding_weights,self._x_input)
        
        if self.debug:
            print('h_weights:',h_weights.get_shape())
            
        h_avg = tf.reduce_sum(h_weights * h_lookup,axis=1) / tf.reduce_sum(h_weights,axis = 1)
        
        if self.debug:
            print('h_avg:',h_avg.get_shape())
        
        
        # reshaping because conv2d expects 4-dim tensors
        h_embed = tf.reshape(h_avg,[PARAMS.batch_size,1,PARAMS.dim_embeddings,1])
        
        if self.debug:
            print('h_embed:',h_embed.get_shape())
            print()
        
        
#        """CONVOLUTIONAL LAYERS"""
#        filter_sizes = PARAMS.gram_sizes
#        nof_features = PARAMS.conv1_feature_count
#        pooled = []
#        for f in filter_sizes:
#            W_conv1f = weight_variable([f,PARAMS.dim_embeddings,1,nof_features],f'W_conv1_{f}')
#            b_conv1f = bias_variable([nof_features],f'b_conv1_{f}')
#            h_conv1f = tf.nn.relu(conv2d(h_embed,W_conv1f) + b_conv1f)
#            h_spp1f = spatial_pyramid_pool(h_conv1f,PARAMS.SPP_dimensions)
#            
#            if self.debug:
#                print(f'h_conv1_f={f}:', h_conv1f.get_shape())
#                print(f'h_spp1_f={f}:', h_spp1f.get_shape())
#                print(f'expected_f={f}:',PARAMS.SPP_output_shape)
#                print('')
#            
#            pooled.append(h_spp1f)
#        
#        h_conv1 =  tf.concat(pooled,axis=1)
#        
#        if self.debug:
#            print('h_conv1:',h_conv1.get_shape())
#            assert h_conv1.get_shape() == (
#                    PARAMS.batch_size,
#                    len(PARAMS.gram_sizes) * PARAMS.SPP_output_shape[1]
#                    )
#            
            
        """FULLY-CONNECTED LAYER with dropout"""
        nof_inputs = PARAMS.dim_embeddings
        nof_neurons = PARAMS.dim_embeddings
        
        W_fc1 = weight_variable([nof_inputs,nof_neurons],'W_fc1')
        b_fc1 = bias_variable([nof_neurons],'b_fc1')
        
        h_in = tf.reshape(h_embed,[PARAMS.batch_size,nof_neurons])
        
        h_fc1 = tf.nn.relu(tf.matmul(h_in,W_fc1) + b_fc1)
        
        if self.debug:
            print('h_fc1:',h_fc1.get_shape())
        
        self._keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob=self._keep_prob)
        
        
        """READOUT LAYER"""
        W_fc2 = weight_variable([nof_neurons,PARAMS.nof_classes],'W_fc2')
        b_fc2 = bias_variable([PARAMS.nof_classes],'b_fc2')
        
        h_fc2 = tf.matmul(h_fc1_drop,W_fc2) + b_fc2
        '''note to self: !! NO RELU HERE !!'''
        
        if self.debug:
            print('h_fc2:',h_fc2.get_shape())
        
        
        """MODEL OUTPUT"""
        y = h_fc2
        self._model = tf.nn.softmax(y)
        
        if self.debug:
            print('y:',y.get_shape())
            assert y.get_shape() == (
                    PARAMS.batch_size,
                    PARAMS.nof_classes
                    )
            print('')
        
        
        """LOSS"""
        loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                        logits=y, labels=self._y_input )
                )
        
        self._train_step = tf.train.AdamOptimizer(PARAMS.learning_rate).minimize(loss)
        
        """ACCURACY"""
        self._correct_predictions = tf.equal(tf.argmax(y,1), tf.argmax(self._y_input,1))
        self._accuracy = tf.reduce_mean(tf.cast(self._correct_predictions, tf.float32))
    
    def _representation(self,tweet):
        tokens = [self.vocab.get(t, -1) for t in tweet.strip().split()]
        tokens = [t for t in tokens if t >= 0]
        return tokens
    
    def _represent(self,*examples,encoding='utf8'):
        if not PARAMS.suppress_output:
            print("representing ...")
        if len(examples) != PARAMS.nof_classes:
            print(f'ERR: expected {PARAMS.nof_classes} classes, got examples for {len(examples)}')
            exit()
            
        tweets = []
        for i in range(len(examples)):
            if self.debug:
                print(f'reading examples for class {i}')
            one_hot = np.zeros([PARAMS.nof_classes])
            one_hot[i] = 1
            
            with open(examples[i], encoding=encoding) as fpos:
                tweets_i = fpos.readlines()
            
            tweets_i_data = zip([self._representation(tweet) for tweet in tweets_i],
                            [one_hot] * len(examples[i])
                            )
            
            tweets += tweets_i_data
        
        '''
        CNN expects all examples in a batch to have the same length
        '''
        tdict = {}
        for t in tweets:
            tdict.setdefault(len(t[0]), []).append(t)
            
        return tdict
    
    def train(self,*examples,encoding='utf8'):
        """the order of the files in examples determines what class they belong to!"""
        
        tdict = self._represent(*examples,encoding=encoding)

        if PARAMS.use_padding:
            self._train_all_padded(tdict)
        else:
            self._train_all(tdict)
    
    def _train_random(self,tdict):
        def next_batch(ex_len,amount=PARAMS.batch_size):
            exs = tdict[ex_len]            
            idxs = nprand.randint(0,len(exs),amount)
            return zip(*[exs[i] for i in idxs ])
        
        """TRAINING"""
        with self._session.as_default():
            for it in range(PARAMS.nof_iterations):
                if it % PARAMS.print_frequency == 0:
                    k = nprand.randint(0,len(tdict.keys()))
                    k = list(tdict.keys())[k]
                    xs,ys = next_batch(k)
                    feed_dict = {self._x_input:xs,
                                 self._y_input:ys,
                                 self._keep_prob:1
                                 }
                    accuracy = self._accuracy.eval(feed_dict=feed_dict)
                    if not PARAMS.suppress_output:
                        print(f'iteration {it}; acc(random sample) = {accuracy}')
                    
                    if self.debug:
                        output = self._session.run(self._model,feed_dict=feed_dict)
                        print('expected',ys)
                        print('got',output)
                        
                    
                top = len(list(tdict.keys())) -1
                curr = 0
                for i in tdict.keys():
                    if not PARAMS.suppress_output:
                        status_update(curr,top,label=f'training size {i}')
                    curr += 1
                    xs,ys = next_batch(i)
                    feed_dict = {self._x_input:xs,
                                 self._y_input:ys,
                                 self._keep_prob:PARAMS.dropout_keep_probability
                                 }
                    self._session.run(self._train_step,feed_dict=feed_dict)
                    
                if not PARAMS.suppress_output:
                    print('saving ...')
                saver = tf.train.Saver(self._tf_variables)
                saver.save(self._session,self._save_as,write_meta_graph=False)
                
    def _train_all(self,tdict):
        """TRAINING"""
        def random_selection(lst,amount=PARAMS.batch_size):
            idxs = nprand.randint(0,len(lst),amount)
            return zip(*[lst[i] for i in idxs ])
        
        top = len(list(tdict.keys()))
        
        with self._session.as_default():
            for epoch in range(PARAMS.nof_iterations):
                if epoch % PARAMS.print_frequency == 0 and not PARAMS.suppress_output:
                    acc = self._test(tdict)
                    print('accuracy(training set) =',acc)
                
                label=f'epoch {epoch}'
                curr = 0
                if not PARAMS.suppress_output:
                    status_update(curr,top,label=label)
                for tpl_lst in tdict.values():
                    xs,ys = zip(*tpl_lst)
                    i = 0
                    while i < len(xs):
                        k = min(i + PARAMS.batch_size,len(xs))
                        batch_xs = list(xs[i:k])
                        batch_ys = list(ys[i:k])
                        
                        missing = (PARAMS.batch_size - len(batch_xs))
                        if missing > 0:
                            miss_xs, miss_ys = random_selection(tpl_lst,amount=missing)
                            batch_xs += miss_xs
                            batch_ys += miss_ys
                            if self.debug:
                                assert len(batch_xs) == PARAMS.batch_size
                                assert len(batch_ys) == PARAMS.batch_size
                                
                        feed_dict = {
                                self._x_input:batch_xs,
                                self._y_input:batch_ys,
                                self._keep_prob:PARAMS.dropout_keep_probability
                                }
                        self._session.run(self._train_step,feed_dict=feed_dict)
                        i += PARAMS.batch_size
                    
                    if not PARAMS.suppress_output:
                        curr += 1
                        status_update(curr,top,label=label)
                    
                if not PARAMS.suppress_output:
                    print('saving ...')
                saver = tf.train.Saver(self._tf_variables)
                saver.save(self._session,self._save_as,write_meta_graph=False)
                
    def _pad(self,tdict):
        """takes a length -> vector mapping and produces a list of equal-length zero-padded vectors"""
        
        if not PARAMS.suppress_output:
            print("padding...")
            
        len_max = np.max(list(tdict.keys()))
        padded = []
        
        for ls in tdict.values():
            len_ls = len(ls[0][0])
            diff = len_max - len_ls
            if diff == 0:
                padded.extend(ls)
            else:
                xs,ys = zip(*ls)
                xs_pad = [np.pad(x,pad_width=(0,diff),mode='constant') for x in xs]
                padded.extend(zip(xs_pad,ys))
                
        assert all(len(x[0]) == len_max for x in padded)
        return padded
                
    def _train_all_padded(self,tdict):
        def random_selection(lst,amount=PARAMS.batch_size):
            idxs = nprand.randint(0,len(lst),amount)
            return zip(*[lst[i] for i in idxs ])
        
        padded = self._pad(tdict)
        len_max = len( padded[0][0])
        
        if not PARAMS.suppress_output:
            print("training...")
            
        top = len(padded)
        
        with self._session.as_default():
            for epoch in range(PARAMS.nof_iterations):
                if epoch % PARAMS.print_frequency == 0 and not PARAMS.suppress_output:
                    acc = self._test({len_max:padded})
                    print('accuracy(training set) =',acc)
                
                label=f'epoch {epoch}'
                curr = 0
                if not PARAMS.suppress_output:
                    status_update(curr,top,label=label)
                    
                nprand.shuffle(padded)
                
                xs,ys = zip(*padded)
                i = 0
                while i < len(xs):
                    k = min(i + PARAMS.batch_size,len(xs))
                    batch_xs = list(xs[i:k])
                    batch_ys = list(ys[i:k])
                    
                    missing = (PARAMS.batch_size - len(batch_xs))
                    if missing > 0:
                        miss_xs, miss_ys = random_selection(padded,amount=missing)
                        batch_xs += miss_xs
                        batch_ys += miss_ys
                        if self.debug:
                            assert len(batch_xs) == PARAMS.batch_size
                            assert len(batch_ys) == PARAMS.batch_size
                            
                    feed_dict = {
                            self._x_input:batch_xs,
                            self._y_input:batch_ys,
                            self._keep_prob:PARAMS.dropout_keep_probability
                            }
                    self._session.run(self._train_step,feed_dict=feed_dict)
                    i += PARAMS.batch_size
                    
                    if not PARAMS.suppress_output:
                        status_update(i,top,label=label)
                    
                if not PARAMS.suppress_output:
                    print('saving ...')
                saver = tf.train.Saver(self._tf_variables)
                saver.save(self._session,self._save_as,write_meta_graph=False)        
    
    def _test(self,tdict):
        nof_hits = 0
        nof_samples = 0
        
        top = len(list(tdict.keys()))
        curr = 0
        label = 'calculating accuracy'
        if not PARAMS.suppress_output:
            status_update(curr,top,label=label)
        for tpl_lst in tdict.values():   
            nprand.shuffle(tpl_lst)
            nof_samples += len(tpl_lst)
            xs,ys = zip(*tpl_lst)
            i = 0
            while i < len(xs):
                k = min(i + PARAMS.batch_size,len(xs))
                batch_xs = list(xs[i:k])
                batch_ys = list(ys[i:k])
                
                missing = (PARAMS.batch_size - len(batch_xs))
                if missing > 0:
                    batch_xs += [ batch_xs[0] ] * missing
                    batch_ys += [ batch_ys[0] ] * missing
                    if self.debug:
                        assert len(batch_xs) == PARAMS.batch_size
                        assert len(batch_ys) == PARAMS.batch_size
            
                nof_hits += self._session.run(
                        tf.reduce_sum(tf.cast(
                                tf.slice(self._correct_predictions,[0],[k-i]),
                                tf.float32)),
                        feed_dict={
                                self._x_input:batch_xs,
                                self._y_input:batch_ys,
                                self._keep_prob:1
                                }
                        )
                i += PARAMS.batch_size
            
            if not PARAMS.suppress_output:
                curr += 1
                status_update(curr,top,label=label)
        
        return nof_hits / nof_samples
                        
    def test(self,*examples,encoding='utf8'):
        '''note to self: do not forget the * when passing examples'''
        
        tdict = self._represent(*examples,encoding=encoding)

        if PARAMS.use_padding:
            padded = self._pad(tdict)
            max_len = len(padded[0][0])
            return self._test({max_len:padded})
        else:
            return self._test(tdict)
    
    def predict(self,tweets):
        tweet_reps = [self._representation(t) for t in tweets]
        nof_tweets = len(tweet_reps)
        
        if(self.debug):
            print(f'predicting {nof_tweets} tweets')
        
        if PARAMS.use_padding:
            if not PARAMS.suppress_output:
                print('predicting with padding')
            max_len = np.max([len(tr) for tr in tweet_reps])
            tweet_reps = [np.pad(tr,pad_width=(0,max_len-len(tr)),mode='constant') for tr in tweet_reps]
            return self._predict(tweet_reps)
        
        else:
            
            predictions = []
            
            """The CNN expects all samples in a batch to have the same length."""
    
            permutation, tweet_reps = zip(*sorted(enumerate(tweet_reps),key=lambda x: len(x[1])))
            i = 0; j = 1
            if not PARAMS.suppress_output:
                status_update(i,nof_tweets,label="Predicting")
            while i < nof_tweets:
                curr_len = len(tweet_reps[i])
                while j < nof_tweets and len(tweet_reps[j]) == curr_len:
                    j += 1
                
                curr_preds = self._predict(tweet_reps[i:j])
                predictions += curr_preds
                
                assert len(curr_preds) == j-i, f"ERR: i,j = {i,j}; expected {j-i} predictions but got {len(curr_preds)}!\ncurr = {curr_preds}\nreps={tweet_reps[i:j]}"
    
                i = j
                j = i + 1
                if not PARAMS.suppress_output:
                    status_update(i,nof_tweets,label="Predicting")
            
            assert len(predictions) == nof_tweets, f"ERR: expected {nof_tweets} predictions but got {len(predictions)}"
            
            inverse_permutation = np.argsort(permutation)
            
            return [ predictions[i] for i in inverse_permutation ]
    
    def _predict(self,tweet_reps):
        nof_tweets = len(tweet_reps)
        predictions = []
        
        i = 0
        do_output = not PARAMS.suppress_output and PARAMS.use_padding
        if do_output:
            top = nof_tweets
            status_update(i,top)
        while i < nof_tweets:
                
            k = min(i + PARAMS.batch_size,nof_tweets)
            batch = list(tweet_reps[i:k])
            
            missing = (PARAMS.batch_size - len(batch))
            if missing > 0:
                batch += [ batch[0] ] * missing
                if self.debug:
                    assert len(batch) == PARAMS.batch_size
        
            preds = self._session.run(
                tf.slice(tf.argmax(self._model,1),[0],[k-i]),
                feed_dict={
                        self._x_input:batch,
                        self._keep_prob:1
                })
                     
            predictions += preds.tolist()     
            i += PARAMS.batch_size
            
            if do_output:
                status_update(i,max([i,top]))
            
        return predictions
    
if __name__ == '__main__':
    datafolder = 'twitter-datasets'
#    train_neg = f'{datafolder}/train_neg.txt'
#    train_pos = f'{datafolder}/train_pos.txt'
    train_neg = f'{datafolder}/train_neg_full.txt'
    train_pos = f'{datafolder}/train_pos_full.txt'

    clf = CNN_TweetClassifier(debug=True,embeddings='embeddingsX_K300_step0.001_epochs50.npy')
    
    
    """the order in which the files are passed to the train/test methods
       determines what class they belong to!"""
    
    
    print("STARTING TRAINING")
    # class 0 if negative, class 1 if positive
    '''uncomment this line when you already have a sufficiently trained model'''
    clf.train(train_neg,train_pos) 

    print("TESTING")
    clf = CNN_TweetClassifier(debug=False,embeddings='embeddingsX_K300_step0.001_epochs50.npy')
    acc = clf.test(train_neg,train_pos)
    print('accuracy on training set:',acc)

#    print("PREDICTING")
#    p = clf.predict(["Hello World"])
#    print('"Hello World" is a positive tweet:','yes' if p[0] == 1 else 'no')

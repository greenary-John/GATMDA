import time
import numpy as np
import tensorflow as tf

from models import GAT
from inits import adj_to_bias
from inits import test_negative_sample
from inits import load_data
from inits import generate_mask
from metrics import masked_accuracy
from metrics import ROC


def train(train_arr, test_arr):
    
    # training params
    batch_size = 1
    nb_epochs = 200
    lr = 0.005  
    l2_coef = 0.0005  
    weight_decay = 5e-4
    hid_units = [8] 
    n_heads = [4, 1] 
    residual = False
    nonlinearity = tf.nn.elu
    model = GAT

    print('----- Opt. hyperparams -----')
    print('lr: ' + str(lr))
    print('l2_coef: ' + str(l2_coef))
    print('----- Archi. hyperparams -----')
    print('nb. layers: ' + str(len(hid_units)))
    print('nb. units per layer: ' + str(hid_units))
    print('nb. attention heads: ' + str(n_heads))
    print('residual: ' + str(residual))
    print('nonlinearity: ' + str(nonlinearity))
    print('model: ' + str(model))

    interaction, features, y_train, y_test, train_mask, test_mask, labels = load_data(train_arr, test_arr)
    nb_nodes = features.shape[0]  
    ft_size = features.shape[1]  

    features = features[np.newaxis]
    interaction = interaction[np.newaxis]
    biases = adj_to_bias(interaction, [nb_nodes], nhood=1) 
    
    nd = np.max(labels[:,0])
    nm = np.max(labels[:,1])
    nd = nd.astype(np.int32)
    nm = nm.astype(np.int32)
    entry_size = nd * nm
    with tf.Graph().as_default():
        with tf.name_scope('input'):
              feature_in = tf.compat.v1.placeholder(dtype=tf.float32, shape=(batch_size, nb_nodes, ft_size))
              bias_in = tf.compat.v1.placeholder(dtype=tf.float32, shape=(batch_size, nb_nodes, nb_nodes))
              lbl_in = tf.compat.v1.placeholder(dtype=tf.int32, shape=(entry_size, batch_size))
              msk_in = tf.compat.v1.placeholder(dtype=tf.int32, shape=(entry_size, batch_size))
              neg_msk = tf.compat.v1.placeholder(dtype=tf.int32, shape=(entry_size,batch_size))
              attn_drop = tf.compat.v1.placeholder(dtype=tf.float32, shape=())
              ffd_drop = tf.compat.v1.placeholder(dtype=tf.float32, shape=())
              is_train = tf.compat.v1.placeholder(dtype=tf.bool, shape=())
        
        final_embedding, coefs = model.encoder(feature_in, nb_nodes, is_train,
                                attn_drop, ffd_drop,
                                bias_mat=bias_in,
                                hid_units=hid_units, n_heads=n_heads,
                                residual=residual, activation=nonlinearity)
        scores = model.decoder(final_embedding, nd)

        loss = model.loss_sum(scores, lbl_in, msk_in, neg_msk, weight_decay, coefs, final_embedding)
    
        accuracy = masked_accuracy(scores, lbl_in, msk_in, neg_msk)
        
        train_op = model.training(loss, lr, l2_coef)

        init_op = tf.group(tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer())

        with tf.compat.v1.Session() as sess:
          sess.run(init_op)

          train_loss_avg = 0
          train_acc_avg = 0

          for epoch in range(nb_epochs):
              
              t = time.time()
              
              ##########    train     ##############
              
              tr_step = 0
              tr_size = features.shape[0] 
              
              neg_mask, label_neg = generate_mask(labels, len(train_arr))
              
              while tr_step * batch_size < tr_size:  
                      _, loss_value_tr, acc_tr = sess.run([train_op, loss, accuracy],
                      feed_dict={
                           feature_in: features[tr_step*batch_size:(tr_step+1)*batch_size],   
                           bias_in: biases[tr_step*batch_size:(tr_step+1)*batch_size],
                           lbl_in: y_train,
                           msk_in: train_mask,
                           neg_msk: neg_mask,
                           is_train: True,
                           attn_drop: 0.1, ffd_drop: 0.1})
                      train_loss_avg += loss_value_tr
                      train_acc_avg += acc_tr
                      tr_step += 1
              print('Epoch: %04d | Training: loss = %.5f, acc = %.5f, time = %.5f' % ((epoch+1), loss_value_tr,acc_tr, time.time()-t))
          
          print("Finish traing.")
          
          ###########     test      ############
          
          ts_size = features.shape[0]
          ts_step = 0
          ts_loss = 0.0
          ts_acc = 0.0
    
          print("Start to test")
          while ts_step * batch_size < ts_size:
              out_come, emb, coef, loss_value_ts, acc_ts = sess.run([scores, final_embedding, coefs, loss, accuracy],
                      feed_dict={
                          feature_in: features[ts_step*batch_size:(ts_step+1)*batch_size],
                          bias_in: biases[ts_step*batch_size:(ts_step+1)*batch_size],
                          lbl_in: y_test,
                          msk_in: test_mask,
                          neg_msk: neg_mask,
                          is_train: False,
                          attn_drop: 0.0, ffd_drop: 0.0})
              ts_loss += loss_value_ts
              ts_acc += acc_ts
              ts_step += 1
          print('Test loss:', ts_loss/ts_step, '; Test accuracy:', ts_acc/ts_step)
              
          out_come = out_come.reshape((nd,nm))
          test_negative_samples = test_negative_sample(labels,len(test_arr),neg_mask.reshape((nd,nm)))
          test_labels, score = ROC(out_come,labels, test_arr,test_negative_samples)  
              
          return test_labels, score
          sess.close()

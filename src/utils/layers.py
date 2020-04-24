import tensorflow as tf
from inits import glorot


conv1d = tf.layers.conv1d
        
def attn_head(seq, out_sz, bias_mat, activation, in_drop=0.0, coef_drop=0.0, residual=False):
  with tf.name_scope('my_attn'):
    if in_drop != 0.0:
       seq = tf.nn.dropout(seq, 1.0 - in_drop)  
    seq_fts = seq
    latent_factor_size = 8  
    
    w_1 = glorot([seq_fts.shape[2].value,latent_factor_size])
    w_2 = glorot([2*seq_fts.shape[2].value,latent_factor_size])
    
    f_1 = tf.layers.conv1d(seq_fts, 1, 1) 
    f_2 = tf.layers.conv1d(seq_fts, 1, 1) 
    logits = f_1 + tf.transpose(f_2, [0, 2, 1])
    coefs = tf.nn.softmax(tf.nn.leaky_relu(logits[0]) + bias_mat[0])
    
    if coef_drop != 0.0:
       coefs = tf.nn.dropout(coefs, 1.0 - coef_drop)
    if in_drop != 0.0:
       seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)
    
    neigh_embs = tf.matmul(coefs, seq_fts[0])
    
    neigh_embs_aggre_1 = tf.matmul(tf.add(seq_fts[0],neigh_embs),w_1)
    neigh_embs_aggre_2 = tf.matmul(tf.concat([seq_fts[0],neigh_embs],axis=-1),w_2)
    
    final_embs = activation(neigh_embs_aggre_1) + activation(neigh_embs_aggre_2)
    
    return final_embs, coefs
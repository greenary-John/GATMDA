import tensorflow as tf

from utils import layers
from models.base_gattn import BaseGAttN
from inits import glorot
from metrics import masked_accuracy


class GAT(BaseGAttN):
    
    def encoder(inputs, nb_nodes, training, attn_drop, ffd_drop,    
            bias_mat, hid_units, n_heads, activation=tf.nn.elu, residual=False):
        attns = []
        for _ in range(n_heads[0]):     
            attn_temp, coefs = layers.attn_head(inputs, bias_mat=bias_mat,      
                out_sz=hid_units[0], activation=activation,   
                in_drop=ffd_drop, coef_drop=attn_drop, residual=False)
            inputs = attn_temp[tf.newaxis]
            attns.append(attn_temp)
        h_1 = tf.concat(attns, axis=-1)   
        return h_1, coefs   
            
    def decoder(embed, nd):
        embed_size = embed.shape[1].value
        with tf.compat.v1.variable_scope("deco"):
             weight3 = glorot([embed_size,embed_size])
        U=embed[0:nd,:]
        V=embed[nd:,:]
        logits=tf.matmul(tf.matmul(U,weight3),tf.transpose(V))
        logits=tf.reshape(logits,[-1,1])
        
        return tf.nn.relu(logits)
    
    def loss_sum(scores, lbl_in, msk_in, neg_msk, weight_decay, coefs, emb):
        loss_basic = masked_accuracy(scores, lbl_in, msk_in, neg_msk)
        
        para_decode = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope="deco")
        loss_basic +=  weight_decay * tf.nn.l2_loss(para_decode)
        
        return loss_basic
    
    
    
    
    
    
    
    
    
    
    
     
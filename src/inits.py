

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import random
import tensorflow as tf
from collections import defaultdict


def adj_to_bias(adj, sizes, nhood=1):       
    nb_graphs = adj.shape[0] 
    mt = np.empty(adj.shape)
    for g in range(nb_graphs):
        mt[g] = np.eye(adj.shape[1])
        for _ in range(nhood):
            mt[g] = np.matmul(mt[g], (adj[g] + np.eye(adj.shape[1])))
        for i in range(sizes[g]):
            for j in range(sizes[g]):
                if mt[g][i][j] > 0.0:
                    mt[g][i][j] = 1.0
    return -1e9 * (1.0 - mt)                            
            
def normalize_features(feat):

    degree = np.asarray(feat.sum(1)).flatten()

    # set zeros to inf to avoid dividing by zero
    degree[degree == 0.] = np.inf
    degree_inv = 1. / degree
    degree_inv_mat = sp.diags([degree_inv], [0])
    feat_norm = degree_inv_mat.dot(feat)

    return feat_norm

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(train_arr, test_arr):
    
    labels = np.loadtxt("data/HMDAD/adj.txt")  
    
    nd = np.max(labels[:,0])
    nm = np.max(labels[:,1])
    nd = nd.astype(np.int32)
    nm = nm.astype(np.int32)
    
    logits_test = sp.csr_matrix((labels[test_arr,2],(labels[test_arr,0]-1, labels[test_arr,1]-1)),shape=(nd,nm)).toarray()
    logits_test = logits_test.reshape([-1,1])  

    logits_train = sp.csr_matrix((labels[train_arr,2],(labels[train_arr,0]-1, labels[train_arr,1]-1)),shape=(nd,nm)).toarray()
    logits_train = logits_train.reshape([-1,1])
      
    train_mask = np.array(logits_train[:,0], dtype=np.bool).reshape([-1,1])
    test_mask = np.array(logits_test[:,0], dtype=np.bool).reshape([-1,1])
    
    M = sio.loadmat('data/HMDAD/interaction.mat')
    M = M['interaction']
     
    interaction = np.vstack((np.hstack((np.zeros(shape=(nd,nd),dtype=int),M)),np.hstack((M.transpose(),np.zeros(shape=(nm,nm),dtype=int)))))      
    
    F1 = np.loadtxt("data/HMDAD/disease_features.txt")
    F2 = np.loadtxt("data/HMDAD/microbe_features.txt") 
    
    features = np.vstack((np.hstack((F1,np.zeros(shape=(F1.shape[0],F2.shape[1]),dtype=int))), np.hstack((np.zeros(shape=(F2.shape[0],F1.shape[0]),dtype=int), F2))))
    features = normalize_features(features)
    return interaction, features, sparse_matrix(logits_train), logits_test, train_mask, test_mask, labels

def generate_mask(labels,N):  
    num = 0
    
    nd = np.max(labels[:,0])
    nm = np.max(labels[:,1])
    nd = nd.astype(np.int32)
    nm = nm.astype(np.int32)
    
    A = sp.csr_matrix((labels[:,2],(labels[:,0]-1, labels[:,1]-1)),shape=(nd,nm)).toarray()
    mask = np.zeros(A.shape)
    label_neg=np.zeros((1*N,2))  
    while(num<1*N):
        a = random.randint(0,nd-1) 
        b = random.randint(0,nm-1) 
        if A[a,b] != 1 and mask[a,b] != 1:
            mask[a,b] = 1
            label_neg[num,0]=a
            label_neg[num,1]=b
            num += 1
    mask = np.reshape(mask,[-1,1])  
    return mask,label_neg

def test_negative_sample(labels,N,negative_mask): 
    num = 0
    (nd,nm)=negative_mask.shape
    A = sp.csr_matrix((labels[:,2],(labels[:,0]-1, labels[:,1]-1)),shape=(nd,nm)).toarray() 
    mask = np.zeros(A.shape)
    test_neg=np.zeros((1*N,2))  
    while(num<1*N):
        a = random.randint(0,nd-1) 
        b = random.randint(0,nm-1) 
        if A[a,b] != 1 and mask[a,b] != 1:
            mask[a,b] = 1
            test_neg[num,0]=a
            test_neg[num,1]=b
            num += 1
    return test_neg

def div_list(ls,n):
    ls_len=len(ls)  
    j = ls_len//n
    ls_return = []  
    for i in range(0,(n-1)*j,j):  
        ls_return.append(ls[i:i+j])  
    ls_return.append(ls[(n-1)*j:])  
    return ls_return

def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    initial = tf.random.uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)

def maxpooling(a):
    a=tf.cast(a,dtype=tf.float32)
    b=tf.reduce_max(a,axis=1,keepdims=True)
    c=tf.equal(a,b)
    mask=tf.cast(c,dtype=tf.float32)
    final=tf.multiply(a,mask)
    ones=tf.ones_like(a)
    zeros=tf.zeros_like(a)
    final=tf.where(final>0.0,ones,zeros)
    return final

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return adj.toarray()

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj)+np.eye(adj.shape[0])
    return adj_normalized
    
def sparse_matrix(matrix):
    sigma = 0.001
    matrix = matrix.astype(np.int32)
    result = np.zeros(matrix.shape)
    for i in range(matrix.shape[0]):
        if matrix[i,0]==0:
           result[i,0]=sigma
        else:
           result[i,0]=1
    return result        

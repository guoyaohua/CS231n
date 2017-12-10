import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  '''
  # 为什么要减去最大值？数值稳定性的解释。#
  在求 exponential 之前将x 的每一个元素减去x_i 的最大值。这样求 exponential 的时候会碰到的最大的数就是 0 了，不会发生overflow 的问题，但是如果其他数原本是正常范围，现在全部被减去了一个非常大的数，于是都变成了绝对值非常大的负数，所以全部都会发生 underflow，但是underflow 的时候得到的是 0，这其实是非常 meaningful 的近似值，而且后续的计算也不会出现奇怪的 NaN。
  当然，总不能在计算的时候平白无故地减去一个什么数，但是在这个情况里是可以这么做的，因为最后的结果要做 normalization，很容易可以证明，这里对x 的所有元素同时减去一个任意数都是不会改变最终结果的——当然这只是形式上，或者说“数学上”，但是数值上我们已经看到了，会有很大的差别。
  '''
  num_classes = W.shape[1]
  num_train = X.shape[0]
  for i in range(num_train):     
      scores = X[i].dot(W)
      #scores -= np.max(scores,axis = 1)
      scores -= np.max(scores)
      #loss_i = -scores[y[i]]+np.log(np.sum(np.exp(scores)))
      loss_i = -np.log(np.exp(scores[y[i]])/np.sum(np.exp(scores)))
      #print(loss_i)
      loss += loss_i
      for j in range(num_classes):
          softmax_output=np.exp(scores[j])/np.sum(np.exp(scores))
          if j == y[i]:
              dW[:,j] += (softmax_output-1)*X[i]
          else:
              dW[:,j] += softmax_output*X[i]
                                    
 # print(loss)
  #print("sss")
  loss /= num_train
  loss += 0.5*reg*np.sum(W*W)
  dW=dW/num_train+reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_classes = W.shape[1]
  num_train = X.shape[0]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  '''
    一定注意一点，函数返回的均为一维向量，一定要reshape
  '''
  scores = X.dot(W)#N by C
  #print(np.max(scores,axis=1))
  #print(np.max(scores,axis=1).reshape(-1,1))  
  shift_scores = scores-np.max(scores,axis=1).reshape(-1,1)
  e_scores = np.exp(shift_scores)# N by C
  output_scores = e_scores/np.sum(e_scores,axis = 1).reshape(-1,1)
  #correct_scores = shift_scores[range(num_train),y]
  #softmax_output = e_scores/np.sum(e_scores,axis=1).reshape(-1,1)
  #print(np.sum(e_scores,axis=1).reshape(-1,1).shape)
  #print(correct_scores)
 # print(np.log(np.sum(e_scores,axis=1).reshape(-1,1)))
  loss_i = -np.log(output_scores[range(num_train),y])
  #loss_i = -correct_scores+np.log(np.sum(e_scores,axis=1))#1 by N
  #loss_i = -correct_scores + np.log(np.sum(e_scores,axis=1).reshape(-1,1))
  #print(loss_i.shape)
  #print(np.sum(loss_i))
  loss = np.sum(loss_i)/num_train+0.5*reg*np.sum(W*W)
  
  dS = output_scores.copy()
  dS[range(num_train),y]-= 1
  #dS = np.exp(shift_scores)/np.sum(e_scores,axis=1).reshape(-1,1)
  #dS[range(num_train),y]-=1
  dW = (X.T).dot(dS)
  dW = dW/num_train+reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


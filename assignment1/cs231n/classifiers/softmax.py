from builtins import range
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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    dscore = np.zeros((X.shape[0],W.shape[1]))
    for i in range(X.shape[0]):
        score = X[i].dot(W) # (10,) # get score 
        score = score-np.max(score) # for numerical stability since exp(score) directly may be too large
        exp_score = np.exp(score) # (10,) exponential 
        norm_score = exp_score[y[i]]/ np.sum(exp_score) # normalization score
        loss += -np.log(norm_score)
        all_norm = exp_score/ np.sum(exp_score)
        all_norm = all_norm.reshape(1,-1)
        dscore[i] = exp_score/ np.sum(exp_score) # (10,1) after normalization, dscore(500, 10)
        dscore[i,y[i]] -= 1

    loss/= X.shape[0]
    loss += reg*np.sum(W*W) # add regularization

    dW = (X.T).dot(dscore) # (3072,10) 
    dW /= X.shape[0]    
    dW += 2 * reg * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    scores = X.dot(W) # (500,10)
    dscore = np.zeros((X.shape[0],W.shape[1])) # (500, 10)
    max_scores = np.max(scores,axis=1).reshape(X.shape[0],-1)
    scores -= max_scores
    exp_scores = np.exp(scores)
    norm_scores = np.divide(exp_scores,np.sum(exp_scores,axis=1).reshape(X.shape[0],-1))
    #print(np.sum(norm_scores[1]))
    #print(norm_scores)
    correct_norm_scores = norm_scores[list(range(X.shape[0])),y]
    loss = np.sum(-np.log(correct_norm_scores))
    dscore = norm_scores
    dscore[list(range(X.shape[0])),y] -=1
    dW = (X.T).dot(dscore) # (3072, 10)
    

    loss /= X.shape[0]
    loss += reg*np.sum(W*W)
    
    dW /= X.shape[0]
    dW += 2 * reg * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW

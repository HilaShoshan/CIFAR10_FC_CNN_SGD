from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dimensions = x.shape  # returns a tuple (N, d_1, ..., d_k)
    N = dimensions[0]
    D = 1  # this will be d_1 * ... * d_k
    for i in range(1, len(dimensions)):  # starting in the second element
        D *= dimensions[i]
    # x = np.reshape(x, (N, D))

    out = np.dot(np.reshape(x, (N, D)), w) + b  # x stays in shape of (N, d_1, ..., d_k)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dimensions = x.shape  # returns a tuple (N, d_1, ..., d_k)
    N = dimensions[0]
    D = 1  # this will be d_1 * ... * d_k
    for i in range(1, len(dimensions)):  # starting in the second element
        D *= dimensions[i]

    dx = np.dot(dout, w.T)
    dx = np.reshape(dx, dimensions)  # return the shape to (N, d_1, ..., d_k)

    dw = np.dot(np.reshape(x, (N, D)).T, dout)

    db = np.sum(dout, axis=0)  # sum the elements in each column in dout matrix

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    out = []
    for num in np.nditer(x):
      if num < 0:
        out.append(0)
      else:
        out.append(num)
    out = np.reshape(out, x.shape)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dx = []
    for i, j in np.nditer([x, dout]):
      if i < 0:
        dx.append(0)
      else:
        dx.append(j)
    dx = np.reshape(dx, x.shape)

    ######################
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx

def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input. 
        

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x_pad, w, b, conv_param)
    """
    out = None
    x_pad = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # padding each cannel at each image input
    pad = conv_param['pad']
    x_pad = np.pad(x, [(0,0), (0,0), (pad, pad), (pad, pad)])

    stride = conv_param['stride']  
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape  

    # reshape w into a 2 dimentional array 
    # such that it will be easier to do convolution with "dot"
    kernel = np.reshape(w, (F, C * HH * WW)).T  
    
    # define output
    H_tag = int(1 + (H + 2 * pad - HH) / stride)
    W_tag = int(1 + (W + 2 * pad - WW) / stride)
    out = np.zeros((N, F, H_tag, W_tag))
    
    # do convolution
    for i in range(H_tag):  # for each output row
      from_row = i * stride
      to_row = from_row + HH
      for j in range(W_tag):  # for each output column
        from_col = j * stride
        to_col = from_col + WW
        x_part = x_pad[:, :, from_row:to_row, from_col:to_col]
        x_part = np.reshape(x_part, (N, C * HH * WW))
        x_conv = x_part.dot(kernel) + b
        out[:, :, i, j] = x_conv

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x_pad, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x_pad, w, b, conv_param = cache 

    pad = conv_param['pad']
    stride = conv_param['stride']
    N, C, H, W = (x_pad.shape[0], x_pad.shape[1], x_pad.shape[2]-2*pad, x_pad.shape[3]-2*pad)
    F, C, HH, WW = w.shape
    H_tag = int(1 + (H + 2 * pad - HH) / stride)
    W_tag = int(1 + (W + 2 * pad - WW) / stride)

    kernel = np.reshape(w, (F, C * HH * WW))
    dkernel = np.zeros((F, C * HH * WW))

    dout_pad = np.zeros((N, C, H+2*pad, W+2*pad))

    for i in range(H_tag):  # for each output row
      from_row = i * stride
      to_row = from_row + HH
      for j in range(W_tag):  # for each output column
        dout_part = dout[:, :, i, j]
        from_col = j * stride
        to_col = from_col + WW
        dout_conv = np.dot(dout_part, kernel).reshape(N, C, HH, WW)
        dout_pad[:, :, from_row:to_row, from_col:to_col] += dout_conv
        x_conv = x_pad[:, :, from_row:to_row, from_col:to_col].reshape(N, C * HH * WW)
        dkernel += dout_part.T.dot(x_conv)

    dx = dout_pad[:, :, pad:-pad, pad:-pad]
    dw = np.reshape(dkernel, (F, C, HH, WW))
    db = np.sum(dout, axis=(0, 2, 3))  # dout.shape = (N, F, HH, WW)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here. Output size is given by 

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W = x.shape
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']

    # define output
    H_tag = int((H - pool_height) / stride + 1)
    W_tag = int((W - pool_width) / stride + 1)
    out = np.zeros((N, C, H_tag, W_tag))

    # do max pooling
    for i in range(H_tag):  # for each output row
      from_row = i * stride
      to_row = from_row + pool_height
      for j in range(W_tag):  # for each output column
        from_col = j * stride
        to_col = from_col + pool_width
        x_part = x[:, :, from_row:to_row, from_col:to_col]
        out[:, :, i, j] = np.max(x_part, axis=(2,3))

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x, pool_param = cache
    N, C, H, W = x.shape
    N, C, H_tag, W_tag = dout.shape
    pool_height = pool_param["pool_height"]
    pool_width = pool_param["pool_width"]
    stride = pool_param["stride"]

    dx = np.zeros((N, C, H, W))

    for i in range(H_tag):  # for each output row
      from_row = i * stride
      to_row = from_row + pool_height
      for j in range(W_tag):  # for each output column
        from_col = j * stride
        to_col = from_col + pool_width
        dout_part = dout[:, :, i, j]
        dout_part = np.reshape(dout_part, N*C)
        x_part = x[:, :, from_row:to_row, from_col:to_col]
        x_part = np.reshape(x_part, (N*C, pool_height*pool_width))
        dx_part = dx[:, :, from_row:to_row, from_col:to_col].reshape((N * C, pool_height*pool_width)).T
        max_indx = np.argmax(x_part, axis=1)  # get the indices of the maximum values along axis 1
        dx_part[max_indx, range(N * C)] += dout_part
        dx_part = np.reshape(dx_part.T, (N, C, pool_height, pool_width))
        dx[:, :, from_row:to_row, from_col:to_col] += dx_part 

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx

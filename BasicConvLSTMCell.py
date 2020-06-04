#!/usr/bin/env python
# coding: utf-8

# In[17]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
Activation=nn.Tanh


# In[3]:


class ConvRNNCell(nn.Module):
    #Abstract Object Representing a Convolutional RNN Cell
    
    def __call__(self, input, stare, scope=None):
        #Run this RNN cell on inputs, starting from given state.
        
        raise NotImplementedError("Abstract Method")
    def state_size(self):
        #size(s) of state(s) used by this cell.
        raise NotImplementedError("Abstract Method")
        
    def output_size(self):
        #Integer or TensorShape: size of outputs produced by this cell."""
        raise NotImplementedError("Abstract method")
    
    def zero_state(self, batch_size, dtype):
        #import pdb;pdb.set_trace()
        """Return zero-filled state tensor(s).
        Args:
          batch_size: int, float, or unit Tensor representing the batch size.
          dtype: the data type to use for the state.
        Returns:
          tensor of shape '[batch_size x shape[0] x shape[1] x num_features]
          filled with zeros
        """
        shape = self.shape 
        num_features = self.num_features
        zeros = torch.zeros([batch_size, shape[0], shape[1], num_features * 2]) 
        return zeros


# In[15]:


class BasicConvLSTMCell(ConvRNNCell):
    #Basic Conv LSTM recurrent network cell
    
    def __init__(self, shape, filter_size, num_features, forget_bias=1.0, input_size=None,
               state_is_tuple=False, activation=Activation):
        """Initialize the basic Conv LSTM cell.
        Args:
          shape: int tuple thats the height and width of the cell
          
          filter_size: int tuple-height & width of the filter
          
          num_features: int-depth of the cell 
          forget_bias: floa-The bias added to forget gates (see above).
          input_size: Deprecated and unused.
          state_is_tuple: If True, accepted and returned states are 2-tuples of
            the `c_state` and `m_state`.  If False, they are concatenated
            along the column axis.  The latter behavior will soon be deprecated.
          activation: Activation function of the inner states.
        """
        
        #if not state_is_tuple:
        #logging.warn(%s: Using a concatenated state is slower and will soon be"
        #              "deprecated. Use state_is_tuple=True.", self)
        
        if input_size is not None:
            logging.warn("%s: The input_size parameter is deprecated.", self)
            
        self.shape=shape
        self.filter_size=filter_size
        self.num_features=num_features
        self._forget_bias=forget_bias
        self._state_is_tuple=state_is_tuple
        self._activation=activation
        
        def state_size(self):
            return (LSTMStateTuple(slef._num_units, self._num_units)
                    if self._state_is_tuple else 2*self._num_units)
        
        def output_size(self):
            return self._num_units
        
        def __call__(self, inputs, state, scope=None):
            """LSTM cell"""
            super(BasicConvLSTMCell, self).__init__()
            
            if self.__state_is_tuple:
                c, h = state
            else:
                c, h = torch.split(state, split_size_or_sections=2, dim=3)
            
            concat = _conv_linear([inputs,h], self.filter_size, self.num_features*4, True) 
            
            i, j, f, o = torch.split(concat, split_size_or_sections=4, dim=3)
            
            new_c = (c * F.sigmoid(f+self_.forget_bias)+F.sigmoid(i)*self._activation(j))
            
            new_h = self._activation(new_c)*F.sigmoid(o)
            
            if self._state_is_tuple:
                new_state = LSTMStateTuple(new_c, new_h)
                    
            else:
                new_state = torch.cat((new_c, new_h), 3)
                
            return new_h, new_state
        

            


# In[20]:


def _conv_linear(args, filter_size, num_features, bias, bias_start=0.0, scope=None):
    """convolution:
    Args:
    args: a 4D Tensor or a list of 4D, batch x n, Tensors.
    filter_size: int tuple of filter height and width.
    num_features: int, number of features.
    bias_start: starting value to initialize the bias; 0 by default.
    scope: VariableScope for the created subgraph; defaults to "Linear".
    Returns:
    A 4D Tensor with shape [batch h w num_features]
    Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """
    #Calculate the total size of arguments on dimension 1
    total_args_size_depth=0
    shapes=[a.get_shape().as_lists() for a in args]
    
    for shape in shapes:
        if len(shapes)!=4:
            raise ValueError("Linear is expecting 4D arguments: %s" %str(shapes))
            
        if not shape[3]:
            raise ValueError("Linear expects shape[4] of arguments: %s" %str(shapes))
            
        else:
            total_arg_size_depth+=shape[3]
        
    dtype=[a.dtype for a in args][0]
    
    matrix=Variable("Matrix",[filter_size[0], filter_size[1], total_arg_size_depth, num_features], dtype=dtype)
    
    if len(args)==1:
        res=F.Conv2d(args[0], matrix, strides=[1,1,1,1], padding='SAME')
    
    else:
        res=F.Conv2d(torch.cat((args),3),matrix, strides=[1,1,1,1],padding='SAME')
        
    if not bias:
        return res
    
    bias_term=Variable("Bias",[num_features], dtype=dtype, initializer=nn.linear(bias_start, dtype=dtype))
    
    return res+bias_term


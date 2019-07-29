import tensorflow as tf
from skimage import morphology


def opening2d(value, kernel, stride=1, padding="SAME"):
    """
    erode and then dilate
    
    Parameters
    ----------
    value : Tensor
        4-D with shape [batch, in_height, in_width, depth].
    kernel : Tensor
        Must have the same type as 'value'. 3-D with shape '[kernel_height, kernel_width, depth]'
    stride : int 
        The stride of the sliding window for the spatial dimensions '[1, 2]' of the input tensor.
    padding : string
        from '"SAME", "VALID"'. The type of padding algorithm to use.    

    Returns
    -------            
    out : tensor
        opened output                
    """
    strides = [1, stride, stride, 1]
    rates = [1, 1, 1, 1]
    out = tf.nn.erosion2d(value, kernel, strides, rates, padding)
    out = tf.nn.dilation2d(out, kernel, strides, rates, padding)
    return out


def closing2d(value, kernel, stride=1, padding="SAME"):
    """
    dilate and then erode.
    
    Parameters
    ----------
    value : 
        4-D with shape [batch, in_height, in_width, depth].
    kernel : 
        Must have the same type as 'value'. 3-D with shape '[kernel_height, kernel_width, depth]'
    stride : int 
        The stride of the sliding window for the spatial dimensions '[1, 2]' of the input tensor.
    padding : string 
        from '"SAME", "VALID"'. The type of padding algorithm to use.

    Returns
    -------            
    out : tensor
        closed output                
    """
    strides = [1, stride, stride, 1]
    rates = [1, 1, 1, 1]
    out = tf.nn.dilation2d(value, kernel, strides, rates, padding)
    out = tf.nn.erosion2d(out, kernel, strides, rates, padding)
    return out


def binary_dilation2d(value, kernel, stride=1, padding="SAME"):
    """
    binary erosion
    
    Parameters
    ----------
    value : tensor
        4-D with shape [batch, in_height, in_width, depth].
    kernel : tensor
        Must have the same type as 'value'. 3-D with shape '[kernel_height, kernel_width, depth]'
    stride : int 
        The stride of the sliding window for the spatial dimensions '[1, 2]' of the input tensor.
    padding : string 
        from '"SAME", "VALID"'. The type of padding algorithm to use.     

    Returns
    -------            
    out : tensor
        dilated output                  
    """
    strides = [1, stride, stride, 1]
    rates = [1, 1, 1, 1]
    output4D = tf.nn.dilation2d(
        value, filter=kernel, strides=strides, rates=rates, padding="SAME"
    )
    output4D = output4D - tf.ones_like(output4D)
    return output4D


def binary_erosion2d(value, kernel, stride=1, padding="SAME"):
    """
    binary erosion
    

    Parameters
    ----------
    value : Tensor 
        4-D with shape [batch, in_height, in_width, depth].
    kernel : tensor 
        Must have the same type as 'value'. 3-D with shape '[kernel_height, kernel_width, depth]'
    stride : int
        The stride of the sliding window for the spatial dimensions '[1, 2]' of the input tensor.
    padding : string
        from '"SAME", "VALID"'. The type of padding algorithm to use.       

    Returns
    -------            
    out : tensor
        eroded output             
    """
    strides = [1, stride, stride, 1]
    rates = [1, 1, 1, 1]
    output4D = tf.nn.erosion2d(
        value, kernel, strides=strides, rates=rates, padding="SAME"
    )
    output4D = output4D + tf.ones_like(output4D)
    return output4D


def binary_closing2d(value, kernel, stride=1, padding="SAME"):
    """
    binary erode and then dilate
    
    Parameters
    ----------
    value : tensor
        4-D with shape [batch, in_height, in_width, depth].
    kernel : tensor
        Must have the same type as 'value'. 3-D with shape '[kernel_height, kernel_width, depth]'
    stride : tensor
        The stride of the sliding window for the spatial dimensions '[1, 2]' of the input tensor.
    padding : tensor
        A 'string' from '"SAME", "VALID"'. The type of padding algorithm to use.

    Returns
    -------            
    out : tensor
        closed output        
    """
    out = binary_dilation2d(value, kernel, stride, padding)
    out = binary_erosion2d(out, kernel, stride, padding)
    return out


def binary_opening2d(value, kernel, stride=1, padding="SAME"):
    """
    binary dilate and then erode

    Parameters
    ----------
    value : tensor. 
        4-D with shape [batch, in_height, in_width, depth].
    kernel : tensor. 
        Must have the same type as 'value'. 3-D with shape '[kernel_height, kernel_width, depth]'
    stride : int. 
        The stride of the sliding window for the spatial dimensions '[1, 2]' of the input tensor.
    padding : string 
        from '"SAME", "VALID"'. The type of padding algorithm to use.

    Returns
    -------            
    out : tensor
        opened output
    """
    out = binary_erosion2d(value, kernel, stride, padding)
    out = binary_dilation2d(out, kernel, stride, padding)
    return out


def square(width, num_kernels):
    """Creates a square shaped structuring element for morphological operations. Internally, scikit-image square is called
    and stacked to the specified number of kernels 'num_kernels'.
    
    Parameters
    ----------
    width : int
        The width and height of the square.
    num_kernels : int
        how many channels the square should have.
    
    Returns
    -------
    kernel : tensor
        kernel shaped [width, height, num_kernels]
    """
    k = tf.convert_to_tensor(morphology.square(width))
    k = tf.stack([k] * num_kernels, axis=-1)
    k = tf.to_int32(k)
    return k


def disk(radius, num_kernels):
    """Generates a flat, disk-shaped structuring element.

    A pixel is within the neighborhood if the euclidean distance between
    it and the origin is no greater than radius.
    Internally, scikit-image disk is called and stacked to the specified number of kernels 'num_kernels'.
    
    Parameters
    ----------
    radius : int
        The radius the disk-shaped structuring element
    num_kernels : int
        how many channels the disk should have. 
    
    Returns
    -------
    kernel : tensor
        kernel shaped [2 * radius + 1, 2 * radius + 1, num_kernels]
    """
    k = tf.convert_to_tensor(morphology.disk(radius))
    k = tf.stack([k] * num_kernels, axis=-1)
    k = tf.to_int32(k)
    return k

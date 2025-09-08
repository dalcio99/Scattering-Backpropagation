import jax
from jax import numpy as jnp
from jax import vmap

# Initialize random params (connectivity matrix J)
def create_conn_matrix(key, kernel_sizes = [6,4], strides = [2,2], input_pixels = 28, num_logits = 10):
    """
    Returns connectivity matrix J

    Args: 
    - num_conv_layers (int)
    - kernel_sizes (List)
    - strides (List)
    - input_pixels (int)
    - num_logits (int)

    Returns:
    - square matrix J
    """

    def create_conv_matrix(key, input_size, output_size, kernel_size, stride):
        # Calculate the number of pixels in input and output images
        input_pixels = input_size * input_size
        output_pixels = output_size * output_size
        
        # Create a matrix to represent the convolution operation (flattened)
        conv_matrix = jnp.zeros((output_pixels, input_pixels))

        indices = jnp.reshape(jnp.array(jnp.meshgrid(jnp.arange(output_size), jnp.arange(input_size), jnp.arange(kernel_size), jnp.arange(kernel_size))), (4, -1))

        def helper(i, j, ki, kj):
            input_x = i * stride + ki
            input_y = j * stride + kj
            input_index = input_x + input_y * input_size
            output_index = i + j * output_size
            
            return output_index, input_index
        
        new_idx = vmap(helper) (*indices)

        conv_matrix = conv_matrix.at[new_idx[0], new_idx[1]].set(1)

        conv_matrix *= jax.random.normal(key, (output_pixels, input_pixels))/ kernel_size

        return conv_matrix


    def create_full_matrix(key, input_size, output_size):
        # Connectivity matrix for a fully connected layer
        return 2 / (input_size + output_size) * jax.random.normal(key, shape = (output_size, input_size))
    
    dimensions = []
    submatrices = []
    


    input_side = input_pixels

    # Create the convolutional (sub)matrices correspondent to each convolutional layer
    # Also store the flattened dimensions into an list
    for kernel_size, stride in zip(kernel_sizes, strides):
        in_dim = input_side**2
        dimensions.append(in_dim)
        output_side = (input_side - kernel_size) // stride + 1
        key, subkey = jax.random.split(key)
        submatrices.append(create_conv_matrix(subkey, input_side, output_side, kernel_size, stride))
        input_side = output_side
    
    dimensions.append(input_side**2)
    dimensions.append(num_logits)
    key, subkey = jax.random.split(key)
    submatrices.append(create_full_matrix(subkey, input_side**2, num_logits))
    block_repr = [[] for _ in range(len(dimensions))]

    # Define J by combining the blocks into a tridiagonal-block matrix
    for k in range(len(dimensions)-1):
        block_repr[k].append(jnp.zeros((dimensions[k], dimensions[k])))
        block_repr[k].append(submatrices[k].T)
        block_repr[k+1].append(submatrices[k])
        for j in range(k+2, len(dimensions)):
            block_repr[k].append(jnp.zeros((dimensions[k], dimensions[j])))
            block_repr[j].append(jnp.zeros((dimensions[j], dimensions[k])))
    block_repr[-1].append(jnp.zeros((dimensions[-1], dimensions[-1])))

    J = jnp.block(block_repr)
    N = J.shape[0]
    key, subkey = jax.random.split(key)

    # Add diagonal frequencies
    J += 1/5 * jnp.diag(jax.random.normal(subkey, shape=(N,)))

    return J
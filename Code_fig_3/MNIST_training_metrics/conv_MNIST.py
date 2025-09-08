import jax
import os
import jax.numpy as jnp
import numpy as np
import tensorflow as tf

from jax import jit, vmap
from functools import partial

os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='.95'

# MNIST Dataset
(input_train, label_train), (input_test, label_test) = tf.keras.datasets.mnist.load_data()
# Create one-hot encoding
num_classes = 10  # MNIST has 10 classes (digits 0-9)
label_train_one_hot = jnp.eye(num_classes)[label_train]
label_test_one_hot = jnp.eye(num_classes)[label_test]

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

# System and loss
@jit
def system(t, y, kappa, kappa1, g, J, x_in, p_in):
    N = len(kappa)
    x = y[:N]
    p = y[N:]
    dxdt = -0.5 * (kappa+kappa1) * x + 0.5 * g * (x**2 + p**2) * p + jnp.dot(J, p) - jnp.sqrt(kappa) * x_in
    dpdt = -0.5 * (kappa+kappa1) * p - 0.5 * g * (x**2 + p**2) * x - jnp.dot(J, x) - jnp.sqrt(kappa) * p_in
    return jnp.concatenate([dxdt, dpdt])

def sigmoid(x):
    return 1/(1 + jnp.exp(-x)) 

def softmax(logits):
    logits -= jnp.max(logits)
    exp_logits = jnp.exp(logits)  # Stabilize softmax
    return exp_logits / jnp.sum(exp_logits)

def cross_entropy_loss(logits, y_true):
    """
    Compute categorical cross-entropy loss.
    
    Args:
    logits  : Model outputs before softmax (shape: [batch_size, num_classes]
    y_true  : One-hot encoded labels (shape: [batch_size, num_classes])
    
    Returns:
    Loss value (scalar)
    """
    temperature = 0.1 
    # When temp < 1 the distribution becomes peaked (more confident)
    # When temp > 1 the distribution becomes softer (more spread out) 
    probs = softmax(logits / temperature)  # Convert logits to probabilities 
    loss = - jnp.sum(y_true * jnp.log(probs + 1e-9)) 
    return loss

cross_entropy_grad = jax.grad(cross_entropy_loss)

def accuracy(x_free, target):
    prediction = jnp.argmax(x_free[-10:])
    return (prediction == jnp.argmax(target))

# Runge-Kutta solver
def rk4_step(f, y, t, dt, kappa, kappa1, g, J, x_in, p_in):
    """Performs a single step of the 4th-order Runge-Kutta method."""
    k1 = f(t, y, kappa, kappa1, g, J, x_in, p_in)
    k2 = f(t + 0.5 * dt, y + 0.5 * dt * k1, kappa, kappa1, g, J, x_in, p_in)
    k3 = f(t + 0.5 * dt, y + 0.5 * dt * k2, kappa, kappa1, g, J, x_in, p_in)
    k4 = f(t + dt, y + dt * k3, kappa, kappa1, g, J, x_in, p_in)
    return y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

def solve_ode(f, y0, t_span, num_steps, dt, kappa, kappa1, g, J, x_in, p_in):
    t_start, t_end = t_span
    ts = jnp.linspace(t_start, t_end, num_steps, dtype=jnp.float32)  # Corrected for inclusion of t_end

    def step(y, t):
        """Step function for JAX scan"""
        y_next = rk4_step(f, y, t, dt, kappa, kappa1, g, J, x_in, p_in)
        return y_next, y_next  # Carry (updated state), output (new state)

    # Run JAX scan for efficient iteration
    _, ys = jax.lax.scan(step, y0, ts)

    return ts, jnp.vstack([y0, ys])  # Stack initial state with results

@partial(jit, static_argnums=(5,))
def update_weights(vec, x_free, p_free, graph, J, N):
    def compute_dFdJ_jl(j, l):
        dFdJ_jl = jnp.zeros(2 * N)  
        dFdJ_jl = dFdJ_jl.at[j].set(p_free[l])
        dFdJ_jl = dFdJ_jl.at[l].set(p_free[j])
        dFdJ_jl = dFdJ_jl.at[j + N].set(-x_free[l])
        dFdJ_jl = dFdJ_jl.at[l + N].set(-x_free[j])
        
        dFdJ_jl = dFdJ_jl.at[j].set(jnp.where(j == l, dFdJ_jl[j] / 2, dFdJ_jl[j]))
        dFdJ_jl = dFdJ_jl.at[j + N].set(jnp.where(j == l, dFdJ_jl[j + N] / 2, dFdJ_jl[j + N]))
    
        return dFdJ_jl
    
    def body_fn(edge):
        j, l = edge
        dFdJ_jl = compute_dFdJ_jl(j, l)
        dJ_jl = jnp.dot(dFdJ_jl, vec)
        return dJ_jl
    
    updates = vmap(body_fn)(graph)

    return updates

def Evaluation_loop(key, J, input_pixels, lower, upper, tmax, num_steps, g):
    """
    Evaluates the model on the test-set portion from 'lower' to 'upper'.
    Args: 
    - key : random key
    - J : parameters (connectivity matrix)
    - input_pixels : total-number of input pixels (784 for MNIST)
    - lower : lowest index to test in the training set (min is 0 for MNIST)
    - upper : highest index to test in the training set (max is 10,000 for MNIST)
    - tmax : final time for dynamic symulation
    - num_steps : number of steps in RK4 (dt = tmax / num_steps)
    - g : nonlinearity strength
    Returns: cross-entropy-loss, accuracy rate
    """
    N = J.shape[0]
    y0 = jax.random.normal(key, shape=(2 * N,))
    kappa = 1. * jnp.ones((N,))
    kappa1 = 1. * jnp.ones((N,))
    loss = 0
    acc_rate = 0
    dt = tmax / num_steps
    
    def evolution(idx, carry):
        y0, J, loss, acc_rate = carry
        # Use dynamic indexing in place of direct indexing
        input_vec = jax.lax.dynamic_index_in_dim(input_test, idx, keepdims=False)
        target = jax.lax.dynamic_index_in_dim(label_test_one_hot, idx, keepdims=False)

        # Encode inputs
        x_in = jnp.zeros(N)
        p_in = jnp.zeros(N)
        x_in = x_in.at[:input_pixels].add(jnp.reshape(input_vec, -1) / 100) # rescale input pixel in the (0, 2.55) interval

        # Inference Phase
        _ , out = solve_ode(system, y0, (0., tmax), num_steps, dt, kappa, kappa1, g, J, x_in, p_in)
        solution_free = out[-1,:]

        x_free = solution_free[:N]
        y0 = solution_free # update initial condition

        # Compute loss for the current sample
        loss += cross_entropy_loss(x_free[-10:], target)
        prediction = jnp.argmax(x_free[-10:])
        acc_rate += (prediction == jnp.argmax(target))
        return y0, J, loss, acc_rate
    
    y0, J, loss, acc_rate = jax.lax.fori_loop(lower, upper, evolution, (y0, J, loss, acc_rate) )
    loss = loss / (upper - lower) 
    acc_rate /= (upper - lower)
    
    return loss, acc_rate


def Training_loop(key, params_history, input_pixels, num_epochs,  batch_size,  beta, learning_rate, lower, upper, tmax, num_steps, g):
    """
    Evaluates the model on the test-set portion from 'lower' to 'upper'.
    Args: 
    - key : random key
    - params_history : list of parameters after each training epoch
    - input_pixels : total-number of input pixels (784 for MNIST)
    - num_epochs : number of training epochs
    - batch_size : used for averaging (approximate) gradients with SGD
    - beta : perturbation strength (for the Feedback Phase)
    - learning_rate : learning rate
    - lower : lowest index to test in the training set (min is 0 for MNIST)
    - upper : highest index to test in the training set (max is 60,000 // batchsize for MNIST)
    - tmax : final time for dynamic symulation
    - num_steps : number of steps in RK4 (dt = tmax / num_steps)
    - g : nonlinearity strength
    Returns: cross-entropy-loss, accuracy rate
    """
    loss_history = [] 
    test_loss_history = []
    acc_history = []
    test_acc_history = []
    J = params_history[0]
    N = J.shape[0]
    
    kappa = 1. * jnp.ones((N,))
    kappa1 = 1. * jnp.ones((N,))
    sigma_x = jnp.block([
    [jnp.zeros((N,N)), jnp.eye(N)],
    [jnp.eye(N), jnp.zeros((N,N))]
    ])
    # Take note of the upper-triangular nonzero entries' indexes
    graph = jnp.stack(jnp.where(J != 0)).T
    dt = tmax / num_steps
    #
    batches = jnp.array([jnp.arange(idx * batch_size, (idx + 1) * batch_size) for idx in range(lower, upper)])

    def evolution(idx, carry):
        epoch_loss, epoch_acc, J = carry

        def single_input_ev(j):
            y0 = jax.random.normal(key, shape=(2 * N,))
            # Use dynamic indexing in place of direct indexing
            input_vec = jax.lax.dynamic_index_in_dim(input_train_shuffled, j, keepdims=False)
            target = jax.lax.dynamic_index_in_dim(label_train_one_hot_shuffled, j, keepdims=False)

            # Encode inputs
            x_in = jnp.zeros(N)
            p_in = jnp.zeros(N)
            x_in = x_in.at[:input_pixels].add(jnp.reshape(input_vec, -1) / 100) # rescale input pixel in the (0, 2.55) interval

            # Inference Phase
            _ , out = solve_ode(system, y0, (0., tmax), num_steps, dt, kappa, kappa1, g, J, x_in, p_in)
            solution_free = out[-1,:]

            x_free = solution_free[:N]
            p_free = solution_free[N:]
            y0 = solution_free 

            # Feedback Phase
            p_in = p_in.at[-10:].add(beta * cross_entropy_grad(x_free[-10:],target) ) # inject error signal
            _ , out = solve_ode(system, y0, (0., tmax/2), num_steps//2, dt, kappa, kappa1, g, J, x_in, p_in)
            solution_perturbed = out[-1,:]

            # Update weights
            # Compute right-part of SB's gradient approx formula 
            # (Note that U' = sigma_x corresponds to the quasi-symmetry U=sigma_y in the a(t)-basis)
            vec = (learning_rate) * sigma_x @ (solution_perturbed - solution_free) / beta 

            return update_weights(vec, x_free, p_free, graph, J, N), cross_entropy_loss(x_free[-10:], target), accuracy(x_free, target)

        updates, losses, accs = vmap(single_input_ev)(batches[idx]) 

        average_updates = jnp.mean(updates, axis = 0)

        J = J.at[graph[:,0], graph[:,1]].add(average_updates)

        epoch_loss += jnp.sum(losses) 
        epoch_acc += jnp.sum(accs)
        
        return epoch_loss, epoch_acc, J
    
    best_acc = 0

    # Training Loop
    for epoch in range(num_epochs): 
        epoch_loss = 0
        epoch_acc = 0
        # Shuffle training set
        key, subkey = jax.random.split(key)
        indices = jax.random.permutation(subkey, upper*batch_size)
        input_train_shuffled = input_train[indices]
        label_train_one_hot_shuffled = label_train_one_hot[indices]
        # Training
        epoch_loss, epoch_acc, J = jax.lax.fori_loop(lower, upper, evolution, (epoch_loss, epoch_acc, J) )

        # Average loss for the epoch
        epoch_loss = epoch_loss / ( (upper - lower) * batch_size )
        epoch_acc = epoch_acc / ( (upper - lower) * batch_size )
        loss_history.append(epoch_loss)
        acc_history.append(epoch_acc)
        # Learning rate update
        ###
        # Compute test accuracy
        key, subkey = jax.random.split(key)       
        epoch_test_loss, epoch_test_acc = Evaluation_loop(key = subkey, J = J, input_pixels = input_pixels, lower = 0, upper = 100, tmax = tmax, num_steps = num_steps, g = g)

        # Save the best weights
        if epoch_test_acc > best_acc:
            best_acc = epoch_test_acc
            params_history[0] = J

        test_loss_history.append(epoch_test_loss)
        test_acc_history.append(epoch_test_acc)
        print(f"End of Epoch {epoch+1}: Train loss = {epoch_loss}, Train Accuracy = {epoch_acc}, Test loss = {epoch_test_loss},  Test Accuracy = {epoch_test_acc}, ||J||_2 = {jnp.linalg.norm(J)}", flush=True)

    return loss_history, acc_history, test_loss_history, test_acc_history, params_history, J

key = jax.random.key(41) 
key, subkey1, subkey2 = jax.random.split(key, num=3)

# Kernel dimension
kernel_sizes = [6,4]
strides = [2,2]

J = create_conn_matrix(subkey1, kernel_sizes = [6,4], strides = [2,2], input_pixels = 28, num_logits = 10)
#print(f"||J||_2 = {jnp.linalg.norm(J)}", flush=True)

params_history = [J]

input_pixels = 784
num_epochs = 400
batch_size = 10
beta = 0.01
learning_rate = 0.1

# Interval of images to use
lower = 0
upper = 600 // batch_size

tmax = 60.
num_steps = 600
g = 0.2

train_loss_history, train_acc_history, test_loss_history, test_acc_history, params_history, J = Training_loop(subkey2, 
                                                                                  params_history, 
                                                                                  input_pixels, 
                                                                                  num_epochs, 
                                                                                  batch_size, 
                                                                                  beta, 
                                                                                  learning_rate, 
                                                                                  lower, 
                                                                                  upper, 
                                                                                  tmax, 
                                                                                  num_steps, 
                                                                                  g)


info = "Mnist Model: \n \
        train_loss_history, train_acc_history, test_loss_history, test_acc_history, kernel_sizes, strides, " \
        "params_history, J, input_pixels, num_epochs, batch_size, beta, \
        learning_rate, lower, upper, tmax, num_steps, g"

data = {"info" : info, 
        "train_loss_history" : train_loss_history,
        "train_acc_history" : train_acc_history,
        "test_loss_history" : test_loss_history,
        "test_acc_history" : test_acc_history,
        "kernel_sizes" : kernel_sizes,
        "strides" : strides,
        "params_history" : params_history, 
        "J": J,
        "input_pixels" : input_pixels, 
        "num_epochs" : num_epochs, 
        "batch_size" : batch_size,
        "beta" : beta, 
        "learning_rate" : learning_rate, 
        "lower" : lower, 
        "upper" : upper, 
        "tmax" : tmax, 
        "num_steps" : num_steps, 
        "g" : g}

np.savez('/DIRECTORY-ADDRESS/data.npz', **data)
#np.save('/DIRECTORY-ADDRESS/data.npy', np.array([data], dtype=object), allow_pickle=True)

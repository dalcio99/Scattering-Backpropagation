import jax
from jax import numpy as jnp

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
    loss = - jnp.sum(y_true * jnp.log(probs + 1e-9)) # Avoid log(0) with small epsilon
    return loss

cross_entropy_grad = jax.grad(cross_entropy_loss)

def accuracy(x_free, target):
    prediction = jnp.argmax(x_free[-10:])
    return (prediction == jnp.argmax(target))
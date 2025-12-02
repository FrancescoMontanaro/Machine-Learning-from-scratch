import numpy as np
from typing import Tuple, List


def _parse_einsum_subscripts(subscripts: str) -> Tuple[List[str], str]:
    """
    Parses einsum subscripts into input and output parts.
    
    Parameters:
    - subscripts (str): Einsum subscript string
    
    Returns:
    - Tuple[List[str], str]: List of input subscripts and output subscript
    """
    
    # Remove spaces
    subscripts = subscripts.replace(" ", "")
    
    # Split into inputs and output
    if "->" in subscripts:
        inputs_str, output = subscripts.split("->")
    else:
        # Implicit output: indices that appear exactly once
        inputs_str = subscripts
        all_indices = "".join(subscripts.split(","))
        output = "".join(sorted(set(c for c in all_indices if all_indices.count(c) == 1)))
    
    # Split input subscripts
    inputs = inputs_str.split(",")
    
    # Return parsed inputs and output
    return inputs, output


def einsum_forward(subscripts: str, *operands: np.ndarray) -> np.ndarray:
    """
    Computes the einsum operation on the given operands.
    
    Parameters:
    - subscripts (str): Einsum subscript string (e.g., 'ij,jk->ik' for matrix multiplication)
    - operands (np.ndarray): Input arrays
    
    Returns:
    - np.ndarray: Result of the einsum operation
    """
    
    # Perform the einsum operation using NumPy
    return np.einsum(subscripts, *operands)


def einsum_backward(
    subscripts: str,
    out_grad: np.ndarray,
    operands: Tuple[np.ndarray, ...],
    operand_index: int
) -> np.ndarray:
    """
    Computes the gradient for a specific operand of the einsum operation.
    
    Parameters:
    - subscripts (str): Einsum subscript string used in forward pass
    - out_grad (np.ndarray): Gradient of the output
    - operands (Tuple[np.ndarray, ...]): All input operands from forward pass
    - operand_index (int): Index of the operand to compute gradient for
    
    Returns:
    - np.ndarray: Gradient for the specified operand
    """
    
    # Parse the subscripts
    inputs, output = _parse_einsum_subscripts(subscripts)
    target_subscript = inputs[operand_index]
    target_shape = operands[operand_index].shape
    
    # Collect all indices available from other operands and output
    available_indices = set(output)
    for i, inp_subscript in enumerate(inputs):
        if i != operand_index:
            available_indices.update(inp_subscript)
    
    # Check which indices in target are not available (summed out indices)
    summed_indices = set(target_subscript) - available_indices
    
    # Check if target subscript has repeated indices (e.g., 'ii' for diagonal)
    unique_indices = []
    repeated_indices = {}
    for idx, char in enumerate(target_subscript):
        if char in unique_indices:
            if char not in repeated_indices:
                repeated_indices[char] = [unique_indices.index(char)]
            repeated_indices[char].append(idx)
        else:
            unique_indices.append(char)
    
    # Handle special cases
    if repeated_indices:
        # Special case: repeated indices in target (e.g., trace: 'ii->')
        unique_target = "".join(unique_indices)
        
        # Build the backward einsum
        backward_inputs = [output]
        backward_operands = [out_grad]
        
        # Collect other inputs and operands
        for i, (inp_subscript, operand) in enumerate(zip(inputs, operands)):
            if i != operand_index:
                backward_inputs.append(inp_subscript)
                backward_operands.append(operand)
        
        # Build the backward subscript
        backward_subscripts = ",".join(backward_inputs) + "->" + unique_target
        
        # Compute the intermediate gradient
        if len(backward_operands) == 1 and backward_operands[0].ndim == 0:
            intermediate_grad = backward_operands[0]
        else:
            intermediate_grad = np.einsum(backward_subscripts, *backward_operands)
        
        # Now expand the repeated indices back to original shape
        grad = np.zeros(target_shape, dtype=out_grad.dtype)
        
        # Fill in the diagonal positions
        if intermediate_grad.ndim == 0:
            n = min(target_shape)
            for idx in range(n):
                grad[idx, idx] = intermediate_grad
        else:
            diag_indices = np.arange(min(target_shape[0], target_shape[-1]))
            grad[diag_indices, diag_indices] = intermediate_grad
        
        return grad
    
    if summed_indices:
        # Special case: some indices were summed out and need to be broadcast back
        # Build the einsum for available indices first
        reduced_target = "".join(c for c in target_subscript if c not in summed_indices)

        # Build the backward einsum
        backward_inputs = [output]
        backward_operands = [out_grad]
        
        # Collect other inputs and operands
        for i, (inp_subscript, operand) in enumerate(zip(inputs, operands)):
            if i != operand_index:
                backward_inputs.append(inp_subscript)
                backward_operands.append(operand)
        
        # Compute the intermediate gradient
        if reduced_target:
            backward_subscripts = ",".join(backward_inputs) + "->" + reduced_target
            intermediate_grad = np.einsum(backward_subscripts, *backward_operands)
        else:
            # All indices were summed - result is scalar
            if len(backward_operands) == 1:
                intermediate_grad = backward_operands[0]
            else:
                backward_subscripts = ",".join(backward_inputs) + "->"
                intermediate_grad = np.einsum(backward_subscripts, *backward_operands)
        
        # Now broadcast to the full target shape
        # Create the shape with 1s for summed dimensions
        broadcast_shape = []
        expand_axes = []
        for idx, char in enumerate(target_subscript):
            if char in summed_indices:
                expand_axes.append(idx)
                broadcast_shape.append(target_shape[idx])
            else:
                broadcast_shape.append(1)
        
        # Expand intermediate gradient and broadcast
        grad = intermediate_grad
        for axis in expand_axes:
            grad = np.expand_dims(grad, axis=axis)
        
        grad = np.broadcast_to(grad, target_shape).copy()
        return grad
    
    # Standard case: no repeated indices in target, no summed indices
    backward_inputs = [output]
    backward_operands = [out_grad]
    
    # Collect other inputs and operands
    for i, (inp_subscript, operand) in enumerate(zip(inputs, operands)):
        if i != operand_index:
            backward_inputs.append(inp_subscript)
            backward_operands.append(operand)
    
    # Build the backward subscript
    backward_subscripts = ",".join(backward_inputs) + "->" + target_subscript
    
    # Compute the gradient
    grad = np.einsum(backward_subscripts, *backward_operands)
    
    return grad
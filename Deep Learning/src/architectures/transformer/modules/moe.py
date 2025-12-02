import numpy as np
from typing import Literal

from ....layers import Dense
from ....activations import SiLU
from ....core import Tensor, Module, ModuleList


class Gate(Module):
    """
    Gating mechanism for Mixture of Experts (MoE) layers.
    """
    
    ### Magic methods ###
    
    def __init__(
        self,
        n_experts: int,
        n_groups: int,
        top_k: int,
        top_k_groups: int,
        score_function: Literal["softmax", "sigmoid"],
        route_scale: float,
        *args, **kwargs
    ) -> None:
        """
        Class constructor for the Gate module.

        Parameters:
        - n_experts: The total number of experts to route to.
        - n_groups: The number of groups to divide the experts into. 
            This is used to hierarchically select experts, by first selecting a group and then selecting experts within it.
        - top_k: The number of top experts to select.
        - top_k_groups: The number of top groups to select.
        - score_function: The scoring function to use for expert selection. Can be either "softmax" or "sigmoid".
        - route_scale: The scaling factor for the routing scores.
        """
        
        # Initialize the parent class
        super().__init__(*args, **kwargs)
        
        # Store parameters
        self.n_groups = n_groups
        self.n_experts = n_experts
        self.top_k = top_k
        self.top_k_groups = top_k_groups
        self.score_function = score_function
        self.route_scale = route_scale
        
        # Initialize the weights and biases for the gating mechanism
        self.weights: Tensor
        self.bias: Tensor
        
        
    ### Protected methods ###

    def _forward_with_multi_outputs(self, x: Tensor, *args, **kwargs) -> tuple[Tensor, Tensor]:
        """
        Foward pass of the layer
        
        Parameters:
        - x (Tensor): Features of the dataset
        
        Returns:
        - tuple[Tensor, Tensor]: Weights and indices of the selected experts
        
        Raises:
        - AssertionError: If the shape of the input data is not valid
        """
        
        # Compute the routing scores
        scores = x @ self.weights # (B, S, E) @ (E, n_experts) -> (B, S, n_experts)
        
        # Apply the score function
        if self.score_function == "softmax":
            # Apply the softmax function to the scores
            scores = scores.softmax(axis=-1)
        elif self.score_function == "sigmoid":
            # Apply the sigmoid function to the scores
            scores = scores.sigmoid()
        else:
            # Invalid score function
            raise ValueError(f"Invalid score function: {self.score_function}. Supported functions are 'softmax' and 'sigmoid'.")
    
        # Store the original scores before adding the bias
        original_scores = scores
        
        # Add the bias to the scores
        scores += self.bias
        
        # If the number of groups is greater than 1, perform hierarchical routing
        # by reshaping expert's scores into groups. At the end, each group will contain
        # n_experts / n_groups experts => n_experts_per_group
        if self.n_groups > 1:
            # Reshape the scores into groups
            scores = scores.reshape((scores.shape[0], self.n_groups, -1)) # (B, S, n_experts) -> (B, S, n_groups, n_experts_per_group)
            
            # Compute the group scores by summing the top k experts per group
            group_scores = scores.top_k(k=2, axis=-1)[0].sum(axis=-1) # (B, S, n_groups, n_experts_per_group) -> (B, S, n_groups)
            
            # Extract the indices of the top k groups
            indices = group_scores.top_k(k=self.top_k_groups, axis=-1)[1] # (B, S, n_groups) -> (B, S, top_k_groups)
            
            # Create a mask for the selected groups
            mask = Tensor(np.ones((x.shape[0], self.n_groups)), dtype=bool).scatter(1, indices, Tensor(np.zeros(indices.shape), dtype=bool)) # (B, n_groups)
            
            # Mask the scores to keep only the selected groups and flatten the last two dimensions
            scores = scores.masked_fill(mask.unsqueeze(-1), float('-inf')).flatten(1) # (B, S, n_groups, n_experts_per_group) -> (B, S, n_experts)
            
        # Extract the top k experts based on the scores
        indices = scores.top_k(k=self.top_k, axis=-1)[1] # (B, S, n_experts) -> (B, S, top_k)
        
        # Gather the original scores for the selected experts
        weights = original_scores.gather(1, indices) # (B, S, n_experts) -> (B, S, top_k)
        
        # If the score function is sigmoid, normalize the weights
        if self.score_function == "sigmoid":
            weights /= weights.sum(axis=-1, keepdims=True)
            
        # Scale the weights
        weights *= self.route_scale
            
        # Return the weights and indices
        return weights, indices
    
    
    def _lazy_init(self, x: Tensor, *args, **kwargs) -> None:
        """
        Method to initialize the module
        
        Parameters:
        - x (Tensor): Features of the dataset
        
        Raises:
        - AssertionError: If the shape of the input data is not valid
        """
        
        # Check if the input shape is valid
        assert len(x.shape) >= 3, f"Invalid input shape. Input must be at least a 3D array. Got shape: {x.shape}"
        
        # Extract the embedding size from the input data
        E = x.shape[-1]
        
        # Initialize the weights for the gating mechanism
        self.weights = Tensor(
            data = np.empty((self.n_experts, E)),
            requires_grad = True,
            is_parameter = True
        )
        
        # Initialize the bias
        # This bias is used to penalize experts that are frequently selected during each training step
        # By adding this bias, the model is encouraged to explore other experts and prevent overfitting to a few experts
        self.bias = Tensor(
            data = np.empty(self.n_experts),
            requires_grad = True,
            is_parameter = True
        )
        

     
class MLP(Module):
    """
    Simple Multi-Layer Perceptron (MLP) for MoE layers.
    """
    
    ### Magic methods ###
    
    def __init__(
        self,
        hidden_dim: int,
        *args, **kwargs
    ) -> None:
        """
        Class constructor for the MLP module.
        
        Parameters:
        - hidden_dim: The number of hidden units in the MLP.
        """
        
        # Initialize the parent class
        super().__init__(*args, **kwargs)
        
        # Store parameters
        self.hidden_dim = hidden_dim
        
        # Initialize the dense layers
        self.w1: Dense # (B, S, E) -> (B, S, hidden_dim)
        self.w2: Dense # (B, S, hidden_dim) -> (B, S, E)
        self.w3: Dense # (B, S, hidden_dim) -> (B, S, E)


    ### Protected methods ###
    
    def _forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        """
        Foward pass of the layer
        
        Parameters:
        - x (Tensor): Features of the dataset
        
        Returns:
        - Tensor: Output of the MLP
        
        Raises:
        - AssertionError: If the shape of the input data is not valid
        """
        
        # Compute and return the output of the MLP
        return self.w2(self.w1(x)) * self.w3(x)
    
    
    def _lazy_init(self, x: Tensor, *args, **kwargs) -> None:
        """
        Method to initialize the module
        
        Parameters:
        - x (Tensor): Features of the dataset
        
        Raises:
        - AssertionError: If the shape of the input data is not valid
        """
        
        # Check if the input shape is valid
        assert len(x.shape) >= 3, f"Invalid input shape. Input must be at least a 3D array. Got shape: {x.shape}"
        
        # Extract the embedding size from the input data
        E = x.shape[-1]
        
        # Initialize the dense layers
        self.w1 = Dense(num_units=self.hidden_dim, activation=SiLU()) # (B, S, E) -> (B, S, hidden_dim)
        self.w2 = Dense(num_units=E) # (B, S, hidden_dim) -> (B, S, E)
        self.w3 = Dense(num_units=self.hidden_dim) # (B, S, hidden_dim) -> (B, S, E)
        
        

class MoE(Module):
    """
    Mixture of Experts (MoE) layer.
    """
    
    ### Magic methods ###
    
    def __init__(
        self,
        n_routed_experts: int,
        n_shared_experts: int,
        n_activated_experts: int,
        mlp_hidden_dim: int,
        *args, **kwargs
    ) -> None:
        """
        Class constructor for the MoE module.
        
        Parameters:
        - n_experts: The total number of experts in the MoE layer.
        - n_activated_experts: The number of experts to activate for each input sample.
        """
        
        # Initialize the parent class
        super().__init__(*args, **kwargs)
        
        # Store parameters
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.n_activated_experts = n_activated_experts
        self.mlp_hidden_dim = mlp_hidden_dim
        
        # Initialize the gate
        self.gate = Gate(
            n_experts = n_routed_experts,
            n_groups = 1,
            top_k = n_activated_experts,
            top_k_groups = 1,
            score_function = "softmax",
            route_scale = 1.0
        )
        
        # Initialize the routed experts
        self.routed_experts = ModuleList([MLP(hidden_dim=mlp_hidden_dim) for _ in range(n_routed_experts)]) 
        
        # Initialize the shared experts
        # In this case, we use a single MLP with output dimension equal to n_shared_experts * mlp_hidden_dim
        self.shared_experts = MLP(hidden_dim=self.n_shared_experts * mlp_hidden_dim)
        
        
    ### Protected methods ###
    
    def _forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        """
        Foward pass of the layer
        
        Parameters:
        - x (Tensor): Features of the dataset
        
        Returns:
        - Tensor: Output of the MoE layer
        """
        
        # Unpack the shape of the input data
        B, S, E = x.shape
        
        # Reshape the input data to combine batch and sequence dimensions
        x = x.reshape((-1, E)) # (B, S, E) -> (B * S, E)
        
        # Compute the gating weights and selected expert indices
        weights, indices = self.gate.forward_with_multi_outputs(x) # (B * S, top_k), (B * S, top_k)
        
        # Compute the output of the routed experts
        counts: list[int] = np.bincount(indices.flatten().to_numpy(), minlength=self.n_routed_experts).tolist()
        
        # Initialize the output tensor
        routed_experts_out = Tensor(np.zeros_like((x.shape))) # (B * S, E)
        
        # Iterate over each expert and compute its output if it was selected
        for i in range(self.n_routed_experts):
            # Skip experts that were not selected
            if counts[i] == 0:
                continue
            
            # Extract the expert
            expert = self.routed_experts[i]
            
            # Get the positions where this expert was selected
            idx, top = np.where(indices.to_numpy() == i)
            
            # Scatter the expert's output to the final output tensor
            routed_experts_out[idx] += expert(x[idx]) * weights[idx, top].unsqueeze(-1)
            
        # Compute the output of the shared experts
        shared_experts_out = self.shared_experts(x) # (B * S, E)

        # Combine the outputs of the routed and shared experts and reshape back
        # to the original batch and sequence dimensions
        out = routed_experts_out + shared_experts_out # (B * S, E)
        out = out.reshape((B, S, E)) # (B * S, E) -> (B, S, E)

        # Return the output
        return out
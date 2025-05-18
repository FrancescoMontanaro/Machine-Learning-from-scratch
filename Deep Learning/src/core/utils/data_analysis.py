import numpy as np
from typing import Optional
import matplotlib.pyplot as plt

from ..tensor import Tensor


def plot_data(datasets: list[dict], title: str, xlabel: str, ylabel: str) -> None:
    """
    Method to plot multiple sets of samples in a 2D space, with options for scatter and line plots.
    
    Parameters:
    - datasets (list[dict]): List of dictionaries, each containing:
        - 'points' (np.ndarray): Dataset to plot with shape (n_samples, 2)
        - 'label' (str): Legend label for the dataset
        - 'color' (str): Color for the dataset plot
        - 'size' (int or float, optional): Size of the points for scatter plot (ignored for line plot)
        - 'plot_type' (str): Type of plot, either 'scatter' or 'line'
    - title (str): Title of the plot
    - xlabel (str): Label of the x-axis
    - ylabel (str): Label of the y-axis
    """
    
    # Create a new figure
    plt.figure(figsize=(10, 6))
    
    # Plot each dataset with its respective color, legend, and style
    for data in datasets:
        points = np.squeeze(data['points'])  # Remove any singleton dimensions
        if points.ndim == 1:  # Reshape to (n_samples, 2) if necessary
            points = points.reshape(-1, 2)
        
        if data['plot_type'] == 'scatter':
            plt.scatter(points[:, 0], points[:, 1], s=data.get('size', 10), c=data['color'], label=data['label'])
        elif data['plot_type'] == 'line':
            # Sort points by the x-axis to ensure a smooth line
            sorted_points = points[points[:, 0].argsort()]
            plt.plot(sorted_points[:, 0], sorted_points[:, 1], color=data['color'], label=data['label'], linewidth=2)
    
    # Set the title and labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    # Add the legend
    plt.legend()
    
    # Show the plot
    plt.show()


def plot_history(train_loss: list[Tensor], valid_loss: Optional[list[Tensor]], title: str, xlabel: str, ylabel: str) -> None:
    """
    Method to plot the training and validation loss
    
    Parameters:
    - train_loss (list[Tensor]): Training losses
    - valid_loss (Optional[list[Tensor]]): Validation losses
    - title (str): Title of the plot
    - xlabel (str): Label of the x-axis
    - ylabel (str): Label of the y-axis
    """
    
    # Create a new figure
    plt.figure(figsize=(8, 4))
    
    # Plot the training and validation loss
    plt.plot(np.array([item.to_numpy() for item in train_loss]), label="Training loss", color="blue")
    if valid_loss: plt.plot(np.array([item.to_numpy() for item in valid_loss]), label="Validation loss", color="orange")
    
    # Set the title and labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    # Add the legend
    plt.legend()
    
    # Show the plot
    plt.show()
 
 
def plot_confusion_matrix(cm: Tensor, title: str = "Confusion Matrix") -> None:
    """
    Method to plot the confusion matrix
    
    Parameters:
    - cm (Tensor): Confusion matrix to plot
    - title (str): Title of the plot
    """
    
    # Create a new figure
    _, ax = plt.subplots(figsize=(6, 6))
    
    # Extract the confusion matrix data
    cm_data = cm.to_numpy()
    
    # Plot the confusion matrix
    ax.matshow(cm_data, cmap=plt.colormaps['Blues'], alpha=0.3)
    
    # Add the class labels
    for i in range(cm_data.shape[0]):
        for j in range(cm_data.shape[1]):
            ax.text(x=j, y=i, s=str(int(cm_data[i, j])), va='center', ha='center')
    
    # Showing the confusion matrix
    plt.xlabel('Predictions', fontsize=12)
    plt.ylabel('Actuals', fontsize=12)
    plt.title(title, fontsize=12)
    plt.show()


def format_summary_output(value: str, width: int) -> str:
    """
    Formats the summary output to fit within a specified width, splitting lines if necessary.
    
    Parameters:
    - value (str): The value to format
    - width (int): The width of the formatted output in characters
    
    Returns:
    - str: The formatted output
    """
    
    # Split the value by spaces to handle word wrapping
    words = value.split()
    formatted_lines = []
    current_line = ""

    # Iterate over the words
    for word in words:
        # Check if adding the word exceeds the width
        if len(current_line) + len(word) + 1 > width:  # +1 for space
            # Add the current line to the list of lines
            formatted_lines.append(current_line)
            current_line = word
        else:
            # Add the word to the current line
            current_line += (" " + word) if current_line else word
    
    # Add the last line
    if current_line:
        formatted_lines.append(current_line)
        
    # Format each line to fit the specified width
    return "\n".join(line.ljust(width) for line in formatted_lines)
import numpy as np
import matplotlib.pyplot as plt


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
    
    
def plot_history(train_loss: np.ndarray, valid_loss: np.ndarray, title: str, xlabel: str, ylabel: str) -> None:
    """
    Method to plot the training and validation loss
    
    Parameters:
    - train_loss (np.ndarray): Training losses
    - valid_loss (np.ndarray): Validation losses
    - title (str): Title of the plot
    - xlabel (str): Label of the x-axis
    - ylabel (str): Label of the y-axis
    """
    
    # Create a new figure
    plt.figure(figsize=(8, 4))
    
    # Plot the training and validation loss
    plt.plot(train_loss, label="Training loss", color="blue")
    plt.plot(valid_loss, label="Validation loss", color="orange")
    
    # Set the title and labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    # Add the legend
    plt.legend()
    
    # Show the plot
    plt.show()
 
 
def plot_confusion_matrix(cm: np.ndarray, title: str = "Confusion Matrix") -> None:
    """
    Method to plot the confusion matrix
    
    Parameters:
    - cm (np.ndarray): Confusion matrix
    - title (str): Title of the plot
    """
    
    # Create a new figure
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Plot the confusion matrix
    ax.matshow(cm, cmap=plt.colormaps['Blues'], alpha=0.3)
    
    # Add the class labels
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(x=j, y=i, s=str(int(cm[i, j])), va='center', ha='center')
    
    # Showing the confusion matrix
    plt.xlabel('Predictions', fontsize=12)
    plt.ylabel('Actuals', fontsize=12)
    plt.title(title, fontsize=12)
    plt.show()
   
    
def one_hot_encoding(y: np.ndarray, n_classes: int) -> np.ndarray:
    """
    Method to perform one-hot encoding on the target variable
    
    Parameters:
    - y (np.ndarray): Target variable
    - n_classes (int): Number of classes
    
    Returns:
    - np.ndarray: One-hot encoded target variable
    """
    
    # Initialize the one-hot encoded target variable
    one_hot = np.zeros((y.shape[0], n_classes))
    
    # Set the appropriate index to 1
    one_hot[np.arange(y.shape[0]), y] = 1
    
    # Return the one-hot encoded target variable
    return one_hot.astype(int)


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


def unbroadcast(arr: np.ndarray, shape: tuple) -> np.ndarray:
    """
    Unbroadcasts an array to a specified shape.
    
    Parameters:
    - arr (np.ndarray): Array to unbroadcast
    - shape (tuple): Shape to unbroadcast to
    
    Returns:
    - np.ndarray: Unbroadcasted array
    """
    
    # If the array has more dimensions than the specified shape, sum along the first axis until the number of dimensions match
    while len(arr.shape) > len(shape):
        arr = arr.sum(axis=0)
        
    # Iterate over the dimensions and sum along the axis if the dimension is 1
    for axis, dim in enumerate(shape):
        # If the dimension is 1, sum along the axis
        if dim == 1 and arr.shape[axis] != 1:
            # Sum along the axis
            arr = arr.sum(axis=axis, keepdims=True)
        
    # Return the unbroadcasted array  
    return arr
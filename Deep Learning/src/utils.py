import numpy as np
import matplotlib.pyplot as plt


def shuffle_data(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Method to shuffle the dataset
    
    Parameters:
    - X (np.ndarray): Features of the dataset
    - y (np.ndarray): Target of the dataset
    
    Returns:
    - tuple[np.ndarray, np.ndarray]: Shuffled features and target
    """
    
    # Get the number of samples
    n_samples = X.shape[0]
    
    # Generate random indices
    indices = np.random.permutation(n_samples)
    
    # Shuffle the dataset
    X_shuffled = X[indices]
    y_shuffled = y[indices]
    
    # Return the shuffled dataset
    return X_shuffled, y_shuffled


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
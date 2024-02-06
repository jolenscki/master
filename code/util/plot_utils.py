import os
import os.path as osp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from google.cloud import storage

def download_blob(bucket_name, source_blob_name, destination_file_name):
    """
    Downloads a blob from a specified bucket in Google Cloud Storage.

    Parameters
    ----------
    bucket_name : str
        The name of the bucket.
    source_blob_name : str
        The name of the blob to download.
    destination_file_name : str
        The name of the file to save the downloaded blob.

    Notes
    -----
    This function requires the `google-cloud-storage` library to be installed.
    """
    # If the file already exists, don't download it again
    if os.path.exists(destination_file_name):
        return

    os.makedirs(os.path.dirname(destination_file_name), exist_ok=True)
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

def download_directory(bucket_name, source_directory, destination_directory):
    """
    Downloads all files from a directory in a Google Cloud Storage bucket to a local directory.
    
    Parameters
    ----------
    bucket_name : str
        The name of the Google Cloud Storage bucket.
    source_directory : str
        The directory path in the bucket to download from.
    destination_directory : str
        The local directory path to download the files to.
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=source_directory)

    for blob in blobs:
        # Skip if blob is a directory
        if blob.name.endswith('/'):
            continue

        local_path = os.path.join(destination_directory, os.path.relpath(blob.name, start=source_directory))
        download_blob(bucket_name, blob.name, local_path)

def upload_file_to_gcs(bucket_name, source_file_name, destination_blob_name):
    """
    Uploads a file to Google Cloud Storage (GCS) bucket.

    Parameters
    ----------
    bucket_name : str
        The name of the GCS bucket.
    source_file_name : str
        The path to the source file to be uploaded.
    destination_blob_name : str
        The name of the destination blob in the GCS bucket.

    Notes
    -----
    This function requires the `google-cloud-storage` library to be installed.

    Examples
    --------
    >>> upload_file_to_gcs("my-bucket", "path/to/file.txt", "destination/file.txt")
    File path/to/file.txt uploaded to destination/file.txt.
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print(f"File {source_file_name} uploaded to {destination_blob_name}.")

def check_dir(dir):
    """
    Create a directory if it doesn't exist.

    Parameters
    ----------
    dir : str
        The directory path.

    Returns
    -------
    None

    Examples
    --------
    >>> check_dir('/path/to/directory')
    """
    if not osp.exists(dir):
        os.makedirs(dir)
        
def loss_per_epoch(
    train_losses,
    epoch,
    specs="specs", 
    save=True, 
    show=True, 
    exp_id="exp",
    ma=False,
    y_scale='log',
    ):
    """
    Plot the training losses per epoch.

    Parameters
    ----------
    train_losses : list
        List of training losses.
    epoch : int
        Number of epochs.
    specs : str, optional
        Specifications for the plot. Defaults to "specs".
    save : bool, optional
        Whether to save the plot. Defaults to True.
    show : bool, optional
        Whether to show the plot. Defaults to True.
    exp_id : str, optional
        Experiment ID. Defaults to "exp".
    ma : bool, optional
        Whether to plot or not the moving average of the series.
    y_scale : str, optional
        Scale of the y-axis. Defaults to 'linear'.
    """
    if epoch == 0:
        epoch = epoch + 1
    # Create a colormap
    cmap = mpl.colormaps.get_cmap('tab10')
    # Create an array of colors
    colors = [cmap(i/epoch) for i in range(epoch)]
    
    plt.figure(figsize=(10, 5))
    ax = plt.gca()
    if ma:
        window = 50
        moving_average = []
        for i in range(len(train_losses) - window + 1):
            moving_average.append(np.mean(train_losses[i:i+window]))
        for i in range(window - 1):
            moving_average.insert(0, np.nan)
        plt.plot(range(len(train_losses)), moving_average, c='indianred', alpha=.7)
    
    # After all epochs
    # Plot training, validation and test losses
    epoch_size = len(train_losses)//epoch
    train_losses = [train_losses[i*epoch_size:(i+1)*epoch_size] for i in range(epoch)]
    fig = plt.gcf()
    for i in range(epoch):
        plt.plot(range(i*len(train_losses[i]), (i+1)*len(train_losses[i])), train_losses[i], c=colors[i], alpha=.5)
        

    
    ax.set_xlabel('Epochs', fontsize=15, color = '#333F4B')
    ax.set_ylabel('Loss', fontsize=15, color = '#333F4B')
    ax.set_yscale(y_scale)
    
    # Set major ticks for epochs
    epoch_ticks = [i*epoch_size for i in range(epoch + 1)]
    epoch_labels = [f'Epoch {i}' for i in range(1, epoch + 1)] + ['']
    plt.xticks(epoch_ticks, epoch_labels)

    
    # change the style of the axis spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if y_scale != 'log':
        ax.set_ylim(bottom=0)
    ax.set_xlim(left=0, right=epoch*epoch_size)
    
    ax.spines['left'].set_position(('outward', 8))
    ax.spines['bottom'].set_position(('outward', 8))
    # Save the figure before showing it
    if save:
        check_dir(osp.join('figures', exp_id))
        plt.tight_layout()
        plt.savefig(osp.join('figures', exp_id, f'loss_{specs}.pdf'), dpi=100, format='pdf')
        plt.clf()
    else:
        plt.show()
    
    return plt.gcf()
        
def loss_per_try(
    train_losses_dict, 
    epoch, 
    specs="specs",
    variable=r'$K_{\text{cheb}}$', 
    save=True, 
    show=True,
    exp_id="exp"
    ):
    """
    Plot the training losses for each try.

    Parameters
    ----------
    train_losses_dict : dict
        A dictionary containing the training losses for each try.
    epoch : int
        The number of epochs.
    specs : str, optional
        The specifications of the plot. Default is "specs".
    variable : str, optional
        The variable to be plotted. Default is '$K_{\text{cheb}}$'.
    save : bool, optional
        Whether to save the plot. Default is True.
    show : bool, optional
        Whether to show the plot. Default is True.
    exp_id : str, optional
        The experiment ID. Default is "exp".
    """
    if epoch == 0:
        epoch = 1
    # Create a colormap
    cmap = mpl.colormaps.get_cmap('tab10')
    # Create an array of colors
    num_tries = len(train_losses_dict)
    colors = [cmap(i/num_tries) for i in range(num_tries)]

    # Initialize the plot
    plt.figure(figsize=(10, 5))
    ax = plt.gca()

    # Plot the training losses for each try
    for i, (key, losses) in enumerate(train_losses_dict.items()):
        batches = range(len(losses))
        plt.plot(batches, losses, color=colors[i], label=f'{key}')

    # Set labels and title
    ax.set_xlabel('Epochs', fontsize=15, color='#333F4B')
    ax.set_ylabel('Loss', fontsize=15, color='#333F4B')

    # Set major ticks for epochs
    epoch_size = len(losses)//epoch
    epoch_ticks = [i*epoch_size for i in range(epoch + 1)]
    epoch_labels = [f'Epoch {i}' for i in range(1, epoch + 1)] + ['']
    plt.xticks(epoch_ticks, epoch_labels)

    # Style adjustments
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_position(('outward', 8))
    ax.spines['bottom'].set_position(('outward', 8))
    
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0, right=epoch*epoch_size)

    # Add legend
    plt.legend(title=variable)

    # Save or show the plot
    if save:
        check_dir(osp.join('figures', exp_id))
        plt.tight_layout()
        plt.savefig(osp.join('figures', exp_id, f'all_loss_{specs}.pdf'), dpi=100, format='pdf')
        plt.clf()
    else:
        plt.show()
        
def reconstruction_plot(
    x, 
    x_recons, 
    specs="specs", 
    save=True, 
    show=True, 
    exp_id="exp"
    ):
    """
    Plots the original and reconstructed data for each channel.

    Parameters
    ----------
    x : torch.Tensor
        The original data.
    x_recons : torch.Tensor
        The reconstructed data.
    specs : str, optional
        The specifications for the plot. Default is "specs".
    save : bool, optional
        Whether to save the plot. Default is True.
    show : bool, optional
        Whether to show the plot. Default is True.
    exp_id : str, optional
        The experiment ID. Default is "exp".
    """
    fig, axs = plt.subplots(1, 2, figsize=(20, 8))  # Create 2 subplots side-by-side

    # Plot original and reconstructed data for channel 0
    xcpu = x.cpu().detach().numpy()
    xrcpu = x_recons.cpu().detach().numpy()
    
    # change the style of the axis spines
    for i, ax in enumerate(axs):
        ax.plot( xcpu[:, 0, i], label='$X$', alpha=.5, c='r')
        ax.plot(xrcpu[:, 0, i], label='$\hat{X}$', alpha=.5, c='dodgerblue')
        ax.set_title(f'Channel {i}')

        ax.set_xlabel('Nodes', fontsize=15, color = '#333F4B')
        ax.set_ylabel('Value', fontsize=15, color = '#333F4B')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim(bottom=0, top=1)
        ax.set_xlim(left=0, right=x.shape[0])
        ax.tick_params(axis='both', which='major', labelsize=15)
    
        ax.spines['left'].set_position(('outward', 8))
        ax.spines['bottom'].set_position(('outward', 8))
        
    handles, labels = plt.gca().get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', fontsize=15, ncols=2)
    
    if save:
        plt.savefig(osp.join('figures', exp_id, f'reconstruction_{specs}.pdf'), dpi=100, format='pdf')
        plt.clf()
    else:
        plt.show()
        
def reconstruction_error_heatmap(
    x, 
    x_recons, 
    specs="specs", 
    save=True, 
    show=True, 
    exp_id="exp"
    ):
    '''
    Calculate and plot the reconstruction error heatmap.

    Parameters
    ----------
    x : torch.Tensor
        The original data, of shape (batch_size, seq_len, height, width, num_channels).
    x_recons : torch.Tensor
        The reconstructed data, of shape (batch_size, seq_len, height, width, num_channels).
    specs : str, optional
        The specifications for the plot. Default is "specs".
    save : bool, optional
        Whether to save the plot. Default is True.
    show : bool, optional
        Whether to show the plot. Default is True.
    exp_id : str, optional
        The experiment ID. Default is "exp".
    '''
    x = x.cpu().detach().numpy()
    x_recons = x_recons.cpu().detach().numpy()
    # Calculate MAE for both channels
    mae = np.abs(x - x_recons)
    avg_mae_speed = np.mean(mae[...,0], axis=(0, 1))
    avg_mae_volume = np.mean(mae[...,1], axis=(0, 1))
    
    # Create figure and axes
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 3.5), dpi=300)

    # Plot heatmaps
    im1 = axs[0].imshow(avg_mae_speed, cmap='Reds')
    im2 = axs[1].imshow(avg_mae_volume, cmap='Reds')

    # Adjust axis spines and ticks
    for ax in axs:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, labelbottom=False, labelleft=False)
    
    axs[0].set_xlabel('Speed', fontsize=15, color='#333F4B')
    axs[1].set_xlabel('Volume', fontsize=15, color='#333F4B')
    
    # Create colorbar
    fig.colorbar(im1, ax=axs.ravel().tolist(), location='right')
    
    if save:
        check_dir(osp.join('figures', exp_id))
        plt.savefig(osp.join('figures', exp_id, f'reconstruction_{specs}.pdf'), dpi=100, format='pdf')
        plt.clf()
    else:
        plt.show()
        
def plot_losses_boxplot(
    losses, 
    variable, 
    errors=['MAE', 'MSE'], 
    specs="",
    save=False, 
    show=False, 
    exp_id="exp"
    ):
    '''
    Plot a boxplot of losses for different experiments.

    Parameters
    ----------
    losses: dict
        A dictionary containing the losses for each experiment.
    variable: str
        The variable being analyzed.
    errors: list, optional
        A list of error metrics to be plotted. Defaults to ['MAE', 'MSE'].
    specs: str, optional
        Additional specifications for the plot. Defaults to "".
    save: bool, optional
        Whether to save the plot as a PDF file. Defaults to False.
    show: bool, optional
        Whether to display the plot. Defaults to False.
    exp_id: str, optional
        The experiment ID. Defaults to "exp".
    '''
    # Create a colormap
    cmap = mpl.colormaps.get_cmap('Pastel1')
    # Create an array of colors
    colors = [cmap(i/len(losses)) for i in range(len(losses))]

    hatches = ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']

    # Create a new figure and a subplot with two y axes
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    for i, (metric, ax, position) in enumerate(zip(errors, (ax1, ax2), (-.2, .2))):
        bp = ax.boxplot(
            [losses[k][metric].cpu().numpy() for k in losses.keys()],
            vert=True,
            positions=np.array(range(1, len(losses) + 1, 1)) + position,
            patch_artist=True,
            widths=.25,
            medianprops={'color': 'darkslategrey', 'linewidth': 2}
            )
        ax.set_ylabel(metric)
        ax.tick_params(axis='y')
        for (patch, color) in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_hatch(hatches[i % len(hatches)]*3)

        # change the style of the axis spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_position(('outward', 4))

        ax.spines['left'].set_position(('outward', 4))
        yticks = [ax.get_yticklabels()[1].get_position()[1], ax.get_yticklabels()[-2].get_position()[1]]
        if i == 0:
            ax.spines['left'].set_bounds(yticks[0], yticks[1])
        else:
            ax.spines['right'].set_bounds(yticks[0], yticks[1])

        ax.spines['bottom'].set_position(('outward', 4))
        ax.spines['bottom'].set_bounds(1, len(losses))
        ax.set_xlabel(variable)

    patches = [mpatches.Patch(facecolor=colors[0],
                              edgecolor='k',
                              hatch=hatches[i % len(hatches)]*3,
                              label=errors[i]) for i in range(len(errors))
              ]

    plt.legend(handles=patches)

    ax1.set_xticks(range(1, len(losses) + 1, 1), labels=losses.keys())
    ax1.xaxis.set_tick_params(rotation=70)
    
    plt.tight_layout()
    if save:
        check_dir(osp.join('figures', exp_id))
        plt.savefig(osp.join('figures', exp_id, f'boxplot_{specs}.pdf'), dpi=200, format='pdf')
        plt.clf()
    else:
        plt.show()
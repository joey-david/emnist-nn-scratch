from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from neural_network import main as nn_main

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

def plot_training_progress(costs, tr_accs, test_accs, times, hyperparameter, value):
    """
    Plot training metrics for neural network training progress.
    Returns a single figure with combined plots for cost, training accuracy, and time per epoch.
    Text positioning has been adjusted to place time/epoch in top left and test accuracy in top right.
    """
    # Create figure with 1 subplot
    fig, ax = plt.subplots(figsize=(15, 5))
    epochs = range(1, len(costs) + 1)
    
    # Plot Training Cost
    ax.plot(epochs, costs, 'b-', label='CE Cost')
    
    # Plot Training Accuracy
    ax.plot(epochs, tr_accs, 'g-', label='Train. Accu.')
    
    # Plot Time per Epoch - now in top left
    avg_time_per_epoch = sum(times) / len(epochs)
    ax.text(0.02, 0.98, f'Avg Time/Epoch: {avg_time_per_epoch:.2f}s', 
             horizontalalignment='left',
             verticalalignment='top',
             transform=ax.transAxes,
             bbox=dict(facecolor='white', alpha=0.8))

    # Set titles and labels
    ax.set_title(f'Training Progress\n({hyperparameter}={value})')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Metrics')
    ax.grid(True)
    
    ax.legend(loc='lower left')
    
    # Add test accuracy as text in top right (position unchanged)
    ax.text(0.98, 0.98, f'Test Acc: {test_accs:.2f}%', 
             horizontalalignment='right',
             verticalalignment='top',
             transform=ax.transAxes,
             bbox=dict(facecolor='white', alpha=0.8))

    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    return fig, (value, tr_accs[-1], test_accs, sum(times))


def compile_graphs(groups):
    """
    Displays rows of graphs in a single png, with each row representing a hyperparameter variation.
    Text positioning has been preserved from the plot_training_progress function.
    
    Parameters:
    groups: list of tuples, each containing (list_of_figures, parameter_name).
            Each figure is a matplotlib figure with 3 subplots.
    """
    num_rows = len(groups)
    num_cols = max(len(figs) for figs, _ in groups)
    subgraph_width = 5
    subgraph_height = 5
    total_width = num_cols * subgraph_width + (num_cols - 1) * 0.3  # Total width for all columns
    total_height = num_rows * subgraph_height + (num_rows - 1) * 0.5  # Total height for all rows
    
    main_fig = plt.figure(figsize=(total_width, total_height))  # Create main figure

    gs = main_fig.add_gridspec(num_rows, 1, height_ratios=[1]*num_rows, hspace=0.5)  # GridSpec for row layout
    
    for row, (figs, param_name) in enumerate(groups):  # Iterate over rows
        row_subplot = main_fig.add_subplot(gs[row])
        row_subplot.axis('off')  # Hide axis for container
        
        # Set row label as parameter name
        row_subplot.text(-0.05, 0.5, param_name, rotation=90, transform=row_subplot.transAxes,
                 fontsize=12, fontweight='bold', verticalalignment='center')
        main_fig.subplots_adjust(right=0.85)  # Add padding to the right
        
        inner_gs = gs[row].subgridspec(1, num_cols, width_ratios=[1] * num_cols, wspace=0.3)
        
        for col in range(num_cols):  # Copy each figure into the grid
            if col < len(figs):
                fig = figs[col]
                orig_axes = fig.get_axes()
                new_ax = main_fig.add_subplot(inner_gs[0, col])
                
                for orig_ax in orig_axes:
                    # Copy line plots
                    for line in orig_ax.get_lines():
                        new_ax.plot(line.get_xdata(), line.get_ydata(), color=line.get_color(),
                                    linestyle=line.get_linestyle(), marker=line.get_marker())
                    
                    # Set grid
                    new_ax.grid(any(line.get_visible() for line in orig_ax.get_xgridlines()) or
                                any(line.get_visible() for line in orig_ax.get_ygridlines()))
                    
                    # Copy labels and title
                    new_ax.set_xlabel(orig_ax.get_xlabel())
                    new_ax.set_ylabel(orig_ax.get_ylabel())
                    new_ax.set_title(orig_ax.get_title())
                    
                    # Copy legend
                    legend = orig_ax.get_legend()
                    if legend:
                        new_ax.legend(handles=legend.legend_handles, 
                                    labels=[text.get_text() for text in legend.get_texts()],
                                    loc='lower left', 
                                    bbox_to_anchor=(0.02, 0.02))
                    
                    # Copy text annotations with adjusted positions
                    for text in orig_ax.texts:
                        text_content = text.get_text()
                        if 'Time/Epoch' in text_content:
                            # Position time/epoch in top left
                            new_ax.text(0.02, 0.98, text_content,
                                      verticalalignment='top',
                                      horizontalalignment='left',
                                      transform=new_ax.transAxes,
                                      bbox=dict(facecolor='white', alpha=0.8))
                        elif 'Test Acc' in text_content:
                            # Position test accuracy in top right
                            new_ax.text(0.98, 0.98, text_content,
                                      verticalalignment='top',
                                      horizontalalignment='right',
                                      transform=new_ax.transAxes,
                                      bbox=dict(facecolor='white', alpha=0.8))

                    # Set aspect ratio
                    new_ax.set_aspect(aspect='auto')
                
                plt.close(fig)  # Close original figure
            else:
                # Create empty subplot
                new_ax = main_fig.add_subplot(inner_gs[0, col])
                new_ax.axis('off')

    main_fig.suptitle('Neural Network Hyperparameter Analysis', fontsize=16, y=0.98)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    output_path = BASE_DIR / f'hyperparameter_analysis_{timestamp}.png'
    main_fig.savefig(output_path, dpi=300, bbox_inches='tight')

def save_weights_and_biases(W0, W1, W2, b0, b1, b2, split, lr, bs, en, nc, nl1, nl2):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    filename = MODELS_DIR / f"weights_biases_S{split}_LR{lr}_BS{bs}_E{en}_NC{nc}_L1{nl1}_L2{nl2}.npz"
    np.savez(str(filename), W0=W0, W1=W1, W2=W2, b0=b0, b1=b1, b2=b2)

def main():
    '''Test the neural network while varying hyperparameters.'''
    #default values
    SPLIT = 0.9
    LEARNING_RATE = 0.002
    BATCH_SIZE = 32
    EPOCH_NUMS = 32
    NUM_CLASSES  = 51
    NEURONS_L1 = 96
    NEURONS_L2 = 64
    
    # Lists to store results for each hyperparameter
    splits = []
    lrs = []
    bss = []
    ens = []
    nls = []
    
    # Training split variation
    print("Varying Training Split")
    for split in np.arange(0.35, 1.0, 0.1):
        split = round(split, 2)
        print(f"Training with split={split}")
        costs, tr_accs, test_accs, times, W0, W1, W2, b0, b1, b2 = nn_main(split, LEARNING_RATE, BATCH_SIZE, EPOCH_NUMS, NUM_CLASSES, NEURONS_L1, NEURONS_L2)
        splits.append(plot_training_progress(costs, tr_accs, test_accs, times, "Training Split", split)[0])
        save_weights_and_biases(W0, W1, W2, b0, b1, b2, split, LEARNING_RATE, BATCH_SIZE, EPOCH_NUMS, NUM_CLASSES, NEURONS_L1, NEURONS_L2)
    
    # Learning rate variation
    print("Varying Learning Rate")
    for lr in np.arange(0.0002, 0.002, 0.0003):
        lr = round(lr, 4)
        print(f"Training with learning rate={lr}")
        costs, tr_accs, test_accs, times, W0, W1, W2, b0, b1, b2 = nn_main(SPLIT, lr, BATCH_SIZE, EPOCH_NUMS, NUM_CLASSES, NEURONS_L1, NEURONS_L2)
        lrs.append(plot_training_progress(costs, tr_accs, test_accs, times, "Learning Rate", lr)[0])
        save_weights_and_biases(W0, W1, W2, b0, b1, b2, SPLIT, lr, BATCH_SIZE, EPOCH_NUMS, NUM_CLASSES, NEURONS_L1, NEURONS_L2)
    
    # Batch size variation
    print("Varying Batch Size")
    for bs in range(5, 100, 15):
        print(f"Training with batch size={bs}")
        costs, tr_accs, test_accs, times, W0, W1, W2, b0, b1, b2 = nn_main(SPLIT, LEARNING_RATE, bs, EPOCH_NUMS, NUM_CLASSES, NEURONS_L1, NEURONS_L2)
        bss.append(plot_training_progress(costs, tr_accs, test_accs, times, "Batch Size", bs)[0])
        save_weights_and_biases(W0, W1, W2, b0, b1, b2, SPLIT, LEARNING_RATE, bs, EPOCH_NUMS, NUM_CLASSES, NEURONS_L1, NEURONS_L2)
    
    # Epoch number variation
    print("Varying Epoch Number")
    for en in range(10, 80, 10):
        
        print(f"Training with epoch number={en}")
        costs, tr_accs, test_accs, times, W0, W1, W2, b0, b1, b2 = nn_main(SPLIT, LEARNING_RATE, BATCH_SIZE, en, NUM_CLASSES, NEURONS_L1, NEURONS_L2)
        ens.append(plot_training_progress(costs, tr_accs, test_accs, times, "Epoch Number", en)[0])
        save_weights_and_biases(W0, W1, W2, b0, b1, b2, SPLIT, LEARNING_RATE, BATCH_SIZE, en, NUM_CLASSES, NEURONS_L1, NEURONS_L2)
    
    # Neurons variation
    print("Varying Neurons in L1")
    for nl in range(20, 210, 30):
        print(f"Training with neurons in L1={int(1.5*nl)}")
        costs, tr_accs, test_accs, times, W0, W1, W2, b0, b1, b2 = nn_main(SPLIT, LEARNING_RATE, BATCH_SIZE, EPOCH_NUMS, NUM_CLASSES, int(1.5*nl), nl)
        nls.append(plot_training_progress(costs, tr_accs, test_accs, times, "Neurons in L1", int(1.5*nl))[0])
        save_weights_and_biases(W0, W1, W2, b0, b1, b2, SPLIT, LEARNING_RATE, BATCH_SIZE, EPOCH_NUMS, NUM_CLASSES, int(1.5*nl), nl)
   
    # Define the hyperparameter groups
    groups = [
        (splits, "Training Split"),
        (lrs, "Learning Rate"),
        (bss, "Batch Size"),
        (ens, "Number of Epochs"),
        (nls, "Number of Neurons (L1)")
    ]
    
    compile_graphs(groups)
    
def test_graph_compile():
    # Create 4 random figures based on the plot_training_progress function
    figs = []
    for i in range(9):
        costs = np.random.rand(10).tolist()
        tr_accs = (np.random.rand(10) * 100).tolist()
        test_accs = np.random.rand() * 100
        times = (np.random.rand(10) * 10).tolist()
        hyperparameter = f'Param {i+1}'
        value = np.random.rand()
        
        fig, _ = plot_training_progress(costs, tr_accs, test_accs, times, hyperparameter, value)
        figs.append(fig)
    
    # Arrange figures in four rows
    groups = [
        (figs[:2], "Row 1"),
        (figs[2:5], "Row 2"),
        (figs[5:], "Row 3"),
    ]
    
    # Compile and save the graphs
    compile_graphs(groups)

if __name__ == '__main__':
    main()

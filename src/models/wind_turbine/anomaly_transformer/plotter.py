import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
#from matplotlib.patches import Rectangle
import os

def plot_confusion_with_metrics(
    true_labels, pred_labels,
    fault_event_name, cmap, save_path=None
):
    from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

    cm   = confusion_matrix(true_labels, pred_labels)
    acc  = accuracy_score(true_labels, pred_labels)
    prec = precision_score(true_labels, pred_labels, zero_division=0)
    rec  = recall_score(true_labels, pred_labels, zero_division=0)
    f1   = f1_score(true_labels, pred_labels, zero_division=0)

    fig, ax = plt.subplots(figsize=(5,5))
    ax.imshow(cm, cmap=cmap, vmin=0, vmax=cm.max(), alpha=0.85)
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(['Pred 0','Pred 1'], fontsize=12, fontweight='bold')
    ax.set_yticklabels(['True 0','True 1'], fontsize=12, fontweight='bold')
    ax.set_ylabel('True label', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted label', fontsize=14, fontweight='bold')
    ax.set_title(f"Confusion Matrix — {fault_event_name}",
                 fontsize=16, fontweight='bold', pad=12)

    """
    # draw raised white squares and then the text
    box_size = 0.16     # side length of square (in data units)
    lift      = -0.012    # how far to lift the box up (in data units)
    for i in [0,1]:
        for j in [0,1]:
            # add a white square behind, lifted up
            rect = Rectangle(
                (j - box_size/2, i - box_size/2 + lift),
                box_size, box_size,
                facecolor='white',
                edgecolor='none',
                transform=ax.transData,
                zorder=2
            )
            ax.add_patch(rect)
            # then add the text centered at the original cell center
            ax.text(
                j, i,
                f"{cm[i,j]}",
                ha='center', va='center',
                fontsize=14, fontweight='bold', color='black',
                zorder=3
            )
    """
    for i in [0,1]:
        for j in [0,1]:
            txt = str(cm[i,j])
            # base pad + 0.05 extra per extra digit
            pad = 0.4 + 0.02 * max(0, len(txt)-1)
            ax.text(
                j, i, txt,
                ha='center', va='center',
                fontsize=14, fontweight='bold', color='black',
                bbox=dict(
                    facecolor='white',
                    edgecolor='none',
                    boxstyle=f'square,pad={pad}'
                ),
                zorder=3
            )

    # metrics string below
    metrics_str = (f"Acc: {acc:.2f}    Prec: {prec:.2f}    "
                   f"Rec: {rec:.2f}    F1: {f1:.2f}")
    fig.text(0.5, 0.02, metrics_str,
             ha='center', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)

    filename = f"confusion_matrix_{fault_event_name.replace(' ','_')}.png"
    outpath = filename if save_path is None else f"{save_path}/{filename}"
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"Saved confusion+metrics to {outpath}")

    return fig

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm, LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter

def plot_attention_with_hotspots(series_fused, timestamp=None, idx=None, high_quantile=0.90, mid_quantile=0.85):
    """
    series_fused: [L,L] array or torch.Tensor of attention weights
    idx: which row to highlight in the bar chart (default = middle)
    high_quantile: top‐quantile cutoff for RED hotspots (default 90th percentile)
    mid_quantile: next‐tier cutoff for BLUE hotspots (default 75th percentile)
    """
    # 1) to numpy
    A = series_fused.detach().cpu().numpy() if hasattr(series_fused, 'cpu') else np.array(series_fused)
    L = A.shape[0]

    # 2) zero‐mask
    zero_mask = (A == 0)

    # 3) compute per‐row cutoffs
    thr_high = np.zeros(L)
    thr_mid  = np.zeros(L)
    for i in range(L):
        row = A[i]
        pos = row[row > 0]
        if pos.size:
            logs = np.log10(pos + 1e-44)
            thr_high[i] = 10 ** np.quantile(logs, high_quantile)
            thr_mid[i]  = 10 ** np.quantile(logs, mid_quantile)

    # 4) pick idx
    if idx is None:
        idx = L // 2
    selected_row = A[idx]

    # 5) symlog norm
    pos_all = A[A > 0]
    vmin = pos_all.min() if pos_all.size else 1e-8
    vmax = pos_all.max()
    norm = SymLogNorm(linthresh=vmin, linscale=0.5, vmin=vmin, vmax=vmax)

    # 6) truncated Blues cmap
    base = plt.cm.Blues
    trunc_colors = base(np.linspace(0.03, 1.0, 256))
    blue_cmap = LinearSegmentedColormap.from_list("truncBlues", trunc_colors)

    # 7) draw
    fig, (ax2d, ax1d) = plt.subplots(1, 2, figsize=(14, 5))
    fig.subplots_adjust(wspace=0.35)

    # 7a) 2D heatmap
    im = ax2d.imshow(A, cmap=blue_cmap, norm=norm)
    grey_canvas = np.where(zero_mask, 1.0, np.nan)
    ax2d.imshow(grey_canvas, cmap='Greys', alpha=0.6, vmin=0, vmax=1)

    # overlay tiered hotspots
    for i in range(L):
        for j in range(L):
            v = A[i, j]
            if v <= 0:
                continue
            if v >= thr_high[i]:
                c = 'red'
            elif v >= thr_mid[i]:
                c = '#3182bd'     # medium blue
            else:
                continue
            ax2d.scatter(j, i, s=20, c=c, marker='s', edgecolors='none')

    ax2d.set_title('Attention Matrix', fontsize=14)
    ax2d.set_xlabel('Key position')
    ax2d.set_ylabel('Query position')
    fig.colorbar(im, ax=ax2d, fraction=0.046, pad=0.04, label='Attention weight')

    # 7b) horizontal bar up to idx
    pos = np.arange(idx + 1)
    vals = selected_row[:idx + 1]
    blues = plt.cm.Blues(np.linspace(0.2, 1.0, len(pos)))
    ax1d.barh(pos, vals, color=blues, height=0.8)

    # overlay tiered bars
    for j in pos:
        v = vals[j]
        if v <= 0:
            c = '#dddddd'
        elif v >= thr_high[idx]:
            c = 'red'
        elif v >= thr_mid[idx]:
            c = '#3182bd'
        else:
            continue
        ax1d.barh(j, v, color=c, alpha=1.0, height=0.8)

    ax1d.set_title(f'Timestep {idx}  [ {timestamp} ]', fontsize=14)
    ax1d.set_ylabel('Key position')
    ax1d.set_xlabel('Attention weight')
    ax1d.set_xscale('symlog', linthresh=vmin)
    ax1d.set_xlim(-vmin, vmax * 1.1)
    ax1d.invert_yaxis()
    ax1d.grid(True, which='both', linestyle='--', alpha=0.3)

    # 5 scientific ticks on x-axis
    ticks = np.logspace(np.log10(vmin), np.log10(vmax), num=5)
    ax1d.set_xticks(ticks)
    ax1d.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"$10^{{{int(np.log10(x))}}}$"))

    plt.tight_layout()
    plt.show()


import matplotlib.pyplot as plt
import numpy as np

def plot_window_scores(cri_row, thresh, max_idx, save_path='.'):
    """
    cri_row : 1D array of length L of anomaly scores for one window
    thresh  : scalar threshold
    max_idx : index of the peak anomaly in this window
    """
    L = len(cri_row)
    x = np.arange(L)

    # 1) bar plot in yellow-orange
    plt.figure(figsize=(8,3))
    bars = plt.bar(x, cri_row, color='yellow')

    # 2) yellow dots on any bar exceeding threshold
    exceed = x[cri_row > thresh]
    plt.scatter(exceed, cri_row[cri_row > thresh],
                s=80, c='#FFA500', zorder=3,
                label=f'> thresh ({thresh:.2e})')

    # 3) red dot at the peak index
    plt.scatter([max_idx], [cri_row[max_idx]],
                s=100, c='red', zorder=4,
                label=f'peak timestep : {max_idx}')

    # 4) decorations
    plt.axhline(thresh, color='red', linestyle='--', linewidth=1)
    plt.title("Anomaly Scores in Fault Window", fontsize=14, fontweight='bold')
    plt.xlabel("Timestep", fontsize=12)
    plt.ylabel("Anomaly score", fontsize=12)
    #tick_step = max(1, L // 6)
    #plt.xticks(np.arange(0, L, tick_step))
    xtick_positions = set(np.linspace(0, L-1, 5, dtype=int))
    xtick_positions.update([max_idx, L-1])
    xtick_positions = sorted(set(xtick_positions))
    plt.xticks(xtick_positions)

    # 6) Bold legend labels using LaTeX-like formatting
    legend = plt.legend(loc='upper right', fontsize=10)
    for text in legend.get_texts():
        text.set_fontweight('bold')

    plt.tight_layout()

    # 7) Save if path is given
    save_path = os.path.join(save_path,'window_score_plot')
    plt.savefig(save_path) #dpi=150, bbox_inches='tight')
    print(f"Saved window score plot to {save_path}")

    plt.show()


def plot_and_save_confusion(true_labels, pred_labels, fault_event_name, save_path=None):
    """
    Compute, display, and optionally save a confusion matrix.
    
    Parameters
    ----------
    true_labels : array-like of shape (n_samples,)
        Ground truth binary labels (0 or 1).
    pred_labels : array-like of shape (n_samples,)
        Predicted binary labels (0 or 1).
    fault_event_name : str
        Name/identifier of the fault event, used in the plot title and filename.
    save_path : str, optional
        Directory in which to save the figure. If None, saves to current working directory.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object (so you can further manipulate if needed).
        
    # Example usage:
    # fig = plot_and_save_confusion(all_true, all_pred, "TurbineFault_A", save_path="results")
    # plt.show()
    """
    # 1) Compute confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    tn, fp, fn, tp = cm.ravel()
    
    # 2) Plot
    fig, ax = plt.subplots(figsize=(4,4))
    im = ax.imshow(cm, interpolation='nearest')
    ax.set_title(f"Confusion Matrix\n{fault_event_name}")
    
    # tick labels
    ax.set_xticks([0,1])
    ax.set_yticks([0,1])
    ax.set_xticklabels(['Pred 0','Pred 1'])
    ax.set_yticklabels(['True 0','True 1'])
    
    # label each cell with its count
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]),
                    ha="center", va="center")
    
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    fig.tight_layout()
    
    # 3) Save to disk if a path was given (or save in cwd)
    filename = f"confusion_{fault_event_name.replace(' ','_')}.png"
    outpath = filename if save_path is None else f"{save_path}/{filename}"
    fig.savefig(outpath)
    print(f"Saved confusion matrix to {outpath}")
    
    return fig


def plot_results(self, test_energy, pred, threshold):
        """
        Plot and save results of anomaly detection.

        Parameters:
        - test_energy (numpy array): Anomaly scores or energies from the test set.
        - pred (numpy array): Predicted labels indicating anomalies (1) and normal points (0).
        - threshold (float): Threshold used for anomaly detection.
        """
        path=self.plot_save_path
        
        if not os.path.exists(path):
        	os.makedirs(path)
        	
        print(len(test_energy),len(pred))
        # Calculate the percentile rank of the model's threshold
        percentile_rank = percentileofscore(test_energy, threshold, kind='weak')
        
        plt.figure(figsize=(14, 6))
        
        # Plot Reconstruction Error
        plt.subplot(1, 2, 1)
        plt.plot(test_energy, label='Reconstruction Error')
        plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold ({percentile_rank:.2f}th %ile)')
        plt.title('Reconstruction Error over Test Data')
        plt.xlabel('Data Point Index')
        plt.ylabel('Reconstruction Error')
        plt.legend()
        
        # Annotate with Win_Size at top right corner
        plt.text(1, 1.05, f'[Win_Size: {self.win_size}]', ha='right', va='top', fontsize=12, transform=plt.gca().transAxes)
        
        # Plot Anomalies
        plt.subplot(1, 2, 2)
        plt.plot(test_energy, label='Reconstruction Error')
        plt.plot(np.where(pred, test_energy, np.nan), 'ro', label='Anomalies')
        plt.title('Detected Anomalies')
        plt.xlabel('Data Point Index')
        plt.ylabel('Reconstruction Error')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'anomaly_detection_results.png')
        plt.close()
    
        # Plot test energy
        plt.figure(figsize=(12, 6))
        plt.plot(test_energy, label='Anomaly Energy')
        #plt.axhline(y=np.percentile(test_energy, 97), color='r', linestyle='--')
        plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold ({percentile_rank:.2f}th percentile)')
        plt.plot(np.where(pred, test_energy, np.nan), 'ro', label='Anomaly Points')
        plt.title('Anomaly Detection Energy')
        plt.xlabel('Time')
        plt.ylabel('Energy')
        plt.legend(loc='upper left')
        plt.grid(True)
        plt.tight_layout()

        # Annotate with Win_Size at top right corner
        plt.text(0.99, 0.975, f'Win_Size: {self.win_size}', ha='right', va='top', transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))
        
        # Save the plot
        plt.savefig('anomaly_energy_plot.png')
        plt.close()

        # Plot predicted anomalies
        plt.figure(figsize=(12, 6))
        plt.plot(pred, label='Predicted Anomalies', marker='o', linestyle='')
        plt.title('Detected Anomalies')
        plt.xlabel('Time')
        plt.ylabel('Anomaly Detection')
        plt.legend(loc='center right')
        plt.grid(True)
        plt.tight_layout()

        # Save the plot
        plt.savefig('predicted_anomalies_plot.png')
        plt.close()
        
        # Plot Distribution of Reconstruction Errors
        plt.figure(figsize=(12, 6))
        plt.hist(test_energy, bins=50, alpha=0.75, edgecolor='black')
        plt.axvline(x=threshold, color='r', linestyle='--', label='Threshold')
        plt.title('Distribution of Reconstruction Errors')
        plt.xlabel('Reconstruction Error')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Save the plot
        plt.savefig('reconstruction_error_distribution.png')
        plt.close()

        # Plot Reconstruction Error Over Time
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(test_energy)), test_energy, label='Reconstruction Error')
        #plt.axhline(y=np.percentile(test_energy, 99), color='r', linestyle='--', label='Threshold')
        plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
        plt.title(f'Reconstruction Error Over Time')
        plt.xlabel('Data Point Index over Time')
        plt.ylabel('Reconstruction Error')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Annotate with Win_Size at top right corner
        plt.text(1, 1.05, f'[Win_Size: {self.win_size}]', ha='right', va='top', fontsize=12, transform=plt.gca().transAxes)

        # Save the plot
        plt.savefig('reconstruction_error_over_time.png')
        plt.close()

        # Compare Different Thresholds
        thresholds = np.linspace(min(test_energy), max(test_energy), num=50)
        anomalies_count = [sum(test_energy > t) for t in thresholds]

        plt.figure(figsize=(12, 6))
        plt.plot(thresholds, anomalies_count, marker='o')
        
        # Highlight the current threshold
        
        anomalies = np.sum(pred).astype(int)
        plt.axvline(x=threshold, color='r', linestyle='--', label='Current Threshold')
        plt.scatter([threshold], [anomalies], color='r', zorder=3)
        
        # Calculate the offset for annotation box
        y_offset = (plt.ylim()[1] - plt.ylim()[0]) * 0.012  # 2% of the y-axis range

        # Annotate with a nice box
        plt.annotate(f'{anomalies}', 
             xy=(threshold, anomalies), 
             xytext=(2.5, y_offset), 
             textcoords='offset points',
             #arrowprops=dict(facecolor='black', arrowstyle='->'),
             bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white'))
        
        # Format the current_threshold value
        formatted_threshold = f"{threshold:.1e}"  # Format to one decimal place in scientific notation
        formatted_threshold = str(formatted_threshold).split('e')[0]

        #formatted_threshold = "{:.1f}".format(float(formatted_threshold))  # Convert back to float for plotting

        plt.text(threshold, -max(anomalies_count)*0.094, f'{formatted_threshold}', fontsize=10.2, ha='center')
        
        plt.title('Number of Detected Anomalies at Different Thresholds')
        plt.xlabel('Threshold')
        plt.ylabel('Number of Anomalies')
        #plt.xticks(list(plt.xticks()[0]) + [1.3*1e-5])
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Set the x-axis to use scientific notation
        ax = plt.gca()
        ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='x', scilimits=(-5,-5))
        
        # Save the plot
        plt.savefig('anomalies_vs_threshold.png')
        plt.close()

        print("Plots saved successfully.")

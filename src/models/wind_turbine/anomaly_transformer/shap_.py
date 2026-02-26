import torch
import shap
import numpy as np
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

from model.AnomalyTransformer import AnomalyTransformer
from shap_utils import ModelWrapperForSHAP

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Run SHAP analysis on the Anomaly Transformer Wrapper model.")
    
    parser.add_argument('--dataset_name', type=str, default='Wind Farm A', help="Name of the dataset wind farm")
    parser.add_argument('--model_save_path', type=str, default='./model checkpoints', help="Directory where the model checkpoint is saved.")
    parser.add_argument('--model_', type=str, required=True, choices=['0','1','2','3'], help='anomaly detector model class: 0: generator-bearing 1: gearbox 2: transformer 3: hydraulic')
    parser.add_argument('--data_path', type=str, default='.', help="Directory where anomalous_inputs.pt and background_data.pt are located.")
    parser.add_argument('--plot_dir', type=str, default='shap_plots', help="Directory to save the output plots.")
    
    parser.add_argument('--num_explanations', type=int, default=None, help="Number of anomalous instances to explain.")
    parser.add_argument('--k', type=int, default=40, help="Max features to select.")
    
    return parser.parse_args()

def main():
    """Main driver function for SHAP analysis."""
    args = parse_arguments()
    
    # --- Configuration ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(args.plot_dir):
        os.makedirs(args.plot_dir)
        print(f"Created plot directory: {args.plot_dir}")

    # --- Load Data ---
    print("Loading data for SHAP analysis...")
    anomalous_inputs_path = os.path.join(args.data_path, 'anomalous_inputs.pt')
    background_data_path = os.path.join(args.data_path, 'background_data.pt')
    
    if not os.path.exists(anomalous_inputs_path) or not os.path.exists(background_data_path):
        raise FileNotFoundError("Please generate anomalous_inputs.pt and background_data.pt first.")

    anomalous_inputs_data = torch.load(anomalous_inputs_path, map_location=DEVICE)
    anomalous_inputs = anomalous_inputs_data['anomalous_inputs']
    feature_names = anomalous_inputs_data['feature_names']
    background_data = torch.load(background_data_path, map_location=DEVICE)
    print(f"Loaded {anomalous_inputs.shape[0]} anomalous inputs.")
    print(f"Loaded background data with shape {background_data.shape}.")

    # --- Load Model & Instantiate Wrapper ---
    print("Loading model...")
    checkpoint_path = os.path.join(args.model_save_path, f'{args.dataset_name}{args.model_}_checkpoint1_.pth')
    
    win_size = anomalous_inputs.shape[1]
    feature_count = anomalous_inputs.shape[2]
    
    model = AnomalyTransformer(win_size=win_size, enc_in=feature_count, c_out=feature_count).to(DEVICE)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE), strict=False)
    model.eval()
    print("Model loaded.")
    
    explainer_model = ModelWrapperForSHAP(model, win_size).to(DEVICE)
    explainer_model.eval()

    # --- Initialize SHAP Explainer ---
    print("Initializing SHAP DeepExplainer...")
    explainer = shap.DeepExplainer(explainer_model, background_data)
    print("SHAP Explainer initialized.")

    # --- Calculate SHAP Values ---
    num_explanations = anomalous_inputs.shape[0] if args.num_explanations is None else args.num_explanations 
    inputs_to_explain_subset = anomalous_inputs[:num_explanations]
    print(f"Calculating SHAP values for {inputs_to_explain_subset.shape[0]} instances...")
    shap_values = explainer.shap_values(inputs_to_explain_subset, check_additivity=False)
    print("SHAP values calculated.")

    shap_values_np = np.asarray(shap_values, dtype='float32')
    inputs_np = inputs_to_explain_subset.cpu().numpy()

    def plot_util(summary=True,bar=False):
        model = { '0' : "Generator Bearing", '1' : 'Gearbox', '2' : 'Transformer', '3' : 'Hydraulic' }
        model_name = model[args.model_]
    
        sv2d = shap_values_np.mean(axis=1) # Shape: (n_samples, n_features)
        in2d = inputs_np.mean(axis=1)     # Shape: (n_samples, n_features)

        nonlocal feature_names
        
        def summary_plot():
            # Summary Plot
            plt.figure()
            shap.summary_plot(sv2d, in2d, feature_names=feature_names, max_display=args.k, show=False)
     
            plt.title(f"SHAP Summary Plot for {model_name}", fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(args.plot_dir, f'shap_summary_plot{args.model_}.png'), dpi=150)
            plt.close()
            print("Saved summary plot.")
    
        mean_abs_shap = np.abs(sv2d).mean(axis=0) #np.sum(np.abs(sv2d), axis=0)
        sorted_indices = np.argsort(mean_abs_shap)[::-1]

        top_indices = sorted_indices[:args.k+1]

        feature_to_remove = 'status_type_id'
        indices_to_plot = []
        for index in top_indices:
            if feature_names[index] != feature_to_remove and len(indices_to_plot)<args.k:
                indices_to_plot.append(index)

        feature_names = [feature_names[i] for i in indices_to_plot]
        sv2d = sv2d[:, indices_to_plot]
        in2d = in2d[:, indices_to_plot]
    
        def bar_plot():
            fig, ax = plt.subplots(figsize=(8.5,12))
            ax.barh(feature_names, mean_abs_shap[indices_to_plot], color='#FF0051')
            ax.set_xlabel("mean(|SHAP value|)")
            ax.invert_yaxis()   # so the largest is on top

            # 4) scientific notation on x‑axis
            ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

            # 5) annotate each bar
            for bar in ax.patches:
                w = bar.get_width()
                ax.text(w, bar.get_y() + bar.get_height()/2,
                    f"{w:.2e}", va='center', ha='left', fontsize=10)

            plt.tight_layout()
            plt.savefig(os.path.join(args.plot_dir, f'shap_bar_plot{args.model_}.png'), dpi=150)
            plt.close()
            print("Saved bar plot.")
            
        sensors_pth=os.path.join(args.data_path,'sensors.txt')
        
        with open(sensors_pth, "w") as output:
            output.write(str(feature_names))
            
        print(f'Selected top {args.k} sensors.')
        print(f"Written to {sensors_pth}.")
        
        print("Generating and saving SHAP plots...")
    
        if summary:
            summary_plot() 
        
        if bar:
            bar_plot()   
    
    plot_util()
    
    print("\nSHAP analysis complete.")

if __name__ == '__main__':
    main()

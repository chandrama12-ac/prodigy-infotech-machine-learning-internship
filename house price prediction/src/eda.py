import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_eda(data_path="data/train.csv", output_dir="outputs"):
    """Performs Exploratory Data Analysis and saves plots."""
    print("Starting Exploratory Data Analysis...")
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}. Please place 'train.csv' in the data folder.")
        return
        
    df = pd.read_csv(data_path)
    
    # Select relevant features
    features = ['GrLivArea', 'BedroomAbvGr', 'FullBath', 'SalePrice']
    # Select only existing columns to prevent KeyError
    available_features = [f for f in features if f in df.columns]
    
    if len(available_features) < 2:
        print("Not enough relevant features in dataset for EDA.")
        return
        
    df_selected = df[available_features]
    
    # 1. Correlation Heatmap
    print("Generating Correlation Heatmap...")
    plt.figure(figsize=(8, 6))
    correlation_matrix = df_selected.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap of Selected Features")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"))
    plt.close()
    
    # 2. Scatter Plots
    print("Generating Scatter Plots...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    if 'SalePrice' in df.columns and 'GrLivArea' in df.columns:
        sns.scatterplot(x='GrLivArea', y='SalePrice', data=df_selected, ax=axes[0])
        axes[0].set_title('SalePrice vs GrLivArea')
        
    if 'SalePrice' in df.columns and 'BedroomAbvGr' in df.columns:
        sns.scatterplot(x='BedroomAbvGr', y='SalePrice', data=df_selected, ax=axes[1])
        axes[1].set_title('SalePrice vs BedroomAbvGr')
        
    if 'SalePrice' in df.columns and 'FullBath' in df.columns:
        sns.scatterplot(x='FullBath', y='SalePrice', data=df_selected, ax=axes[2])
        axes[2].set_title('SalePrice vs FullBath')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "scatter_plots.png"))
    plt.close()
    
    # 3. Handle outliers conceptually in EDA (just showing them)
    if 'GrLivArea' in df.columns:
        print("Generating Outlier Boxplot for GrLivArea...")
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=df['GrLivArea'])
        plt.title("Boxplot of GrLivArea (Identifying Outliers)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "outliers_boxplot.png"))
        plt.close()
    
    print(f"EDA completed. Visualizations saved in '{output_dir}/'.")

if __name__ == "__main__":
    run_eda()

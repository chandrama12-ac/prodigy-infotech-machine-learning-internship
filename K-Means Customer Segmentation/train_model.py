import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pickle
import os
import warnings

warnings.filterwarnings('ignore') # ignore some annoying warning messages from seaborn/kmeans

def main():
    print("Starting the Customer Segmentation ML Pipeline...")
    
    # 1. Data Loading
    try:
        df_mall = pd.read_csv("Mall_Customers.csv")
        print("Data loaded successfully! Shape:", df_mall.shape)
    except FileNotFoundError:
        print("Error: Mall_Customers.csv not found in the current directory.")
        return

    print("\n--- First 5 rows ---")
    print(df_mall.head())
    
    # 2. Data Preprocessing
    print("\nChecking for missing values:")
    print(df_mall.isnull().sum())
    # Luckily no missing values in this standard dataset, but good to check anyway
    
    # We only care about Income and Spending Score for this segmentation
    # Drop CustomerID, Gender, Age for now
    X = df_mall.iloc[:, [3, 4]].values 
    
    # Apply feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("Data scaling completed.")
    
    # Create directories for saving outputs
    os.makedirs("plots", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # 3. Exploratory Data Analysis (EDA)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=df_mall['Annual Income (k$)'], y=df_mall['Spending Score (1-100)'], color='teal', alpha=0.7)
    plt.title('EDA: Annual Income vs Spending Score')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig('plots/eda_scatter.png')
    plt.close() # don't show, just save
    print("EDA plot saved to plots/eda_scatter.png")
    
    # 4. Finding Optimal Clusters (Elbow Method)
    wcss = []
    # Test k from 1 to 10
    for i in range(1, 11):
        kmeans_temp = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans_temp.fit(X_scaled)
        wcss.append(kmeans_temp.inertia_)
        
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 11), wcss, marker='o', linestyle='-', color='red')
    plt.title('The Elbow Method')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('WCSS')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig('plots/elbow_method.png')
    plt.close()
    print("Elbow method plot saved to plots/elbow_method.png")
    # From the graph, the elbow is clearly at k=5
    
    # 5. Model Implementation
    optimal_k = 5
    print(f"\nTraining K-Means model with k={optimal_k}...")
    kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42)
    
    # Assign cluster labels 
    cluster_labels = kmeans.fit_predict(X_scaled)
    df_mall['Cluster'] = cluster_labels
    
    # 6. Visualization of the final clusters
    plt.figure(figsize=(10, 7))
    
    colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#c2c2f0']
    cluster_names = ['Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4']
    
    for i in range(optimal_k):
        plt.scatter(X_scaled[cluster_labels == i, 0], X_scaled[cluster_labels == i, 1], 
                    s=100, c=colors[i], label=cluster_names[i], edgecolors='black', alpha=0.8)
        
    # Mark centroids
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
                s=250, c='yellow', label='Centroids', marker='X', edgecolors='black')
                
    plt.title('Customer Segments (K-Means Clustering)')
    plt.xlabel('Annual Income (Scaled)')
    plt.ylabel('Spending Score (Scaled)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.savefig('plots/clusters.png')
    plt.close()
    print("Cluster visualization saved to plots/clusters.png")
    
    # Note on Insights:
    # We will display the exact business logic and meaning of these clusters inside the Streamlit app!
    
    # Save the model and scaler using pickle
    with open('models/kmeans_model.pkl', 'wb') as f:
        pickle.dump(kmeans, f)
        
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
        
    print("\nModel and Scaler successfully saved to the 'models/' directory!")
    print("Pipeline finished.")

if __name__ == '__main__':
    main()

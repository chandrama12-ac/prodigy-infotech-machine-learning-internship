import pandas as pd
import numpy as np
import os

def generate_data():
    """Generates a dummy train.csv mimicking the Kaggle dataset."""
    print("Generating dummy dataset...")
    np.random.seed(42)
    n_samples = 1500
    
    # Generate features
    gr_liv_area = np.random.normal(1500, 500, n_samples)
    bedrooms = np.random.randint(1, 6, n_samples)
    full_bath = np.random.randint(1, 4, n_samples)
    
    # Generate target variable with some noise
    # Base price + sqft + bedrooms + baths + noise
    sale_price = 50000 + (gr_liv_area * 100) + (bedrooms * 10000) + (full_bath * 15000) + np.random.normal(0, 20000, n_samples)
    
    # Add some outliers conceptually
    gr_liv_area[0:10] = 5000
    sale_price[0:10] = 100000 # Unusually low price for huge area
    
    df = pd.DataFrame({
        'Id': range(1, n_samples + 1),
        'GrLivArea': gr_liv_area,
        'BedroomAbvGr': bedrooms,
        'FullBath': full_bath,
        'SalePrice': sale_price,
        # Add some useless columns to simulate the real dataset
        'MSSubClass': np.random.randint(20, 190, n_samples),
        'LotArea': np.random.normal(10000, 2000, n_samples)
    })
    
    # Add missing values as requested
    missing_indices = np.random.choice(n_samples, size=50, replace=False)
    df.loc[missing_indices, 'BedroomAbvGr'] = np.nan
    
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/train.csv', index=False)
    print("Dummy dataset created successfully at data/train.csv")

if __name__ == "__main__":
    generate_data()

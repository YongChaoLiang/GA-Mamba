'''
Oversampling the original training set via Mixup
'''

import pandas as pd
import numpy as np

def extract_dmax(filename):
    df = pd.read_csv(filename)
    labels = df['Dmax']
    cols_to_drop = ['Dmax']
  
    if 'index' in df.columns:
        cols_to_drop.append('index')
    datas = df.drop(columns=cols_to_drop)
  
    return datas, labels
  
def mixup_augmentation(input_path, output_path, threshold=12, n_samples=100, alpha=0.2):
    """
    Mixup-based data augmentation (applied only to minority-class samples).

    Args:
        input_path: Path to the input CSV file.
        output_path: Path to the output CSV file.
        threshold: Dmax threshold used to identify the minority class.
        n_samples: Number of synthetic samples to generate.
        alpha: Beta distribution parameter controlling the interpolation weight λ.
    """
  
    datas, labels = extract_dmax(input_path)
    X = datas.values
    y = labels.values
    mask = (y > threshold)
    X_reset = X.copy()
    y_reset = y.copy()
    minority_X = X_reset[mask]
    minority_y = y_reset[mask]
    minority_count = len(minority_X)
  
    print(f"Original dataset size: {len(datas)}")
    print(f"Original minority-class sample count: {minority_count}")
  
    if minority_count < 2:
        raise ValueError("At least two minority-class samples are required for Mixup")
    synthetic_X = []
    synthetic_y = []
    for _ in range(n_samples):
        idx1, idx2 = np.random.choice(minority_count, size=2, replace=False)
        x1, x2 = minority_X[idx1], minority_X[idx2]
        y1, y2 = minority_y[idx1], minority_y[idx2]
        lam = np.random.beta(alpha, alpha)
        new_x = lam * x1 + (1 - lam) * x2
        new_y = lam * y1 + (1 - lam) * y2
        new_x = np.maximum(new_x, 0) 
        row_sum = new_x.sum()
        if row_sum > 0:
            new_x /= row_sum
        new_x = np.around(new_x, decimals=5) 
        synthetic_X.append(new_x)
        synthetic_y.append(new_y)
    synthetic_X = np.array(synthetic_X)
    synthetic_y = np.array(synthetic_y)
    augmented_X = np.vstack([X_reset, synthetic_X])
    augmented_y = np.hstack([y_reset, synthetic_y])
    resampled_df = pd.DataFrame(augmented_X, columns=datas.columns)
    resampled_df['Dmax'] = augmented_y
    resampled_df['index'] = range(1, len(resampled_df) + 1)
    resampled_df.to_csv(output_path, index=False)
  
    print(f"Augmented dataset size: {len(resampled_df)}")
    print(f"Number of added samples: {n_samples}")
    print(f"Augmented data saved to: {output_path}")
  
    return resampled_df
  
if __name__ == '__main__':
    data_path = 'train_data.csv'
    output_path = 'processed_data/processed_mixup3.csv'
    mixup_augmentation(
        input_path=data_path,
        output_path=output_path,
        threshold=10,
        n_samples=100,
        alpha=0.3  
   )

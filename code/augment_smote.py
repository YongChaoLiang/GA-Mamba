'''
Oversampling the original training set via SMOTE
'''

import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import NearestNeighbors

def extract_dmax(filename):
    df = pd.read_csv(filename)
    labels = df['Dmax']
    cols_to_drop = ['Dmax']
    if 'index' in df.columns:
        cols_to_drop.append('index')
    datas = df.drop(columns=cols_to_drop)
    return datas, labels
  
def smote_augmentation(input_path, output_path, threshold=12, n_samples=100, k_neighbors=5):
    """
    Data augmentation using SMOTE.

    Args:
        input_path: Path to the input CSV file.
        output_path: Path to the output CSV file.
        threshold: Dmax threshold; samples with Dmax > threshold are treated as the minority class.
        n_samples: Number of minority-class samples to generate.
        k_neighbors: Number of neighbors used by the SMOTE algorithm.
    """
  
    datas, labels = extract_dmax(input_path)
    binary_labels = (labels > threshold).astype(int)
    minority_mask = (binary_labels == 1)
    original_minority_count = np.sum(minority_mask)
    original_minority_dmax = labels[minority_mask].values
  
    print(f"Original dataset size: {len(datas)}")
    print(f"Original minority-class sample count:{original_minority_count}")
  
    # Apply SMOTE
    smote = SMOTE(
        sampling_strategy={1: original_minority_count + n_samples},
        k_neighbors=min(k_neighbors, original_minority_count - 1),
        random_state=42
    )
  
    X_resampled, y_resampled = smote.fit_resample(datas.values, binary_labels.values)
    n_original = len(datas)
    synthetic_indices = np.arange(n_original, len(X_resampled))
    for i in synthetic_indices:
        X_resampled[i] = np.maximum(X_resampled[i], 0)
        row_sum = np.sum(X_resampled[i])
        if row_sum > 0:
            X_resampled[i] = X_resampled[i] / row_sum
        X_resampled[i] = np.around(X_resampled[i], decimals=5)
    knn = NearestNeighbors(n_neighbors=min(k_neighbors, original_minority_count))
    knn.fit(datas.values[minority_mask])
    synthetic_dmax = []
    for idx in synthetic_indices:
        distances, indices = knn.kneighbors([X_resampled[idx]], return_distance=True)
        weights = 1 / (distances[0] + 1e-8)  
        weighted_avg = np.average(original_minority_dmax[indices[0]], weights=weights)
        synthetic_dmax.append(weighted_avg)
    resampled_dmax = np.concatenate([
        labels.values,  
        synthetic_dmax  
    ])
  
    resampled_df = pd.DataFrame(X_resampled, columns=datas.columns)
    resampled_df['Dmax'] = resampled_dmax
    resampled_df['index'] = range(1, len(resampled_df) + 1)
    resampled_df.to_csv(output_path, index=False)
  
    print(f"Augmented dataset size:{len(resampled_df)}")
    print(f"Number of added samples: {len(resampled_df) - len(datas)}")
    print(f"Augmented data saved to: {output_path}")
  
    return resampled_df
  
if __name__ == '__main__':
    data_path = 'train_data.csv'
    output_path = 'processed_data/processed_SMOTE.csv'
  
    # Data augmentation using SMOTE
    smote_augmentation(
        input_path=data_path,
        output_path=output_path,
        threshold=10,  
        n_samples=100,  
        k_neighbors=5  
    )

'''
Oversampling the original training set via Gaussian Noise (GN)
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
  
# def add_gaussian_noise(input_path, output_path, threshold=12, noise_level=0.1):
#     datas, labels = extract_dmax(input_path)
#     mask = labels > threshold
#     datas_reset = datas.reset_index(drop=True)
#     labels_reset = labels.reset_index(drop=True)
#     minority_data = datas_reset[mask]
#     minority_labels = labels_reset[mask]
#     noise = np.random.normal(0, noise_level, minority_data.shape)
#     noisy_data = minority_data + noise
#     noisy_df = pd.DataFrame(noisy_data, columns=datas.columns)
#     resampled_data = pd.concat([datas_reset, noisy_df], ignore_index=True)
#     resampled_labels = pd.concat([labels_reset, minority_labels], ignore_index=True)
#     resampled_df = resampled_data.copy()
#     resampled_df['Dmax'] = resampled_labels
#     resampled_df.reset_index(drop=True, inplace=True)
#     resampled_df.to_csv(output_path, index=False)
#     print(f"Augmented dataset size: {len(resampled_df)}")
#     print(f"Number of added samples: {len(resampled_df) - len(datas)}")
#     print(f"Augmented data saved to: {output_path}")
#     return resampled_data, resampled_labels

# Improved version
def add_gaussian_noise(input_path, output_path, threshold=12, noise_level=0.1):
    datas, labels = extract_dmax(input_path)
    mask = labels > threshold
    datas_reset = datas.reset_index(drop=True)
    labels_reset = labels.reset_index(drop=True)
    minority_data = datas_reset[mask]
    minority_labels = labels_reset[mask]
    noisy_data = minority_data.apply(
        lambda row: add_random_noise(row.values, len(row), noise_level),
        axis=1,
        result_type='expand'
    ).to_numpy()
  
    # Normalization: enforce each row to sum to 1
    row_sums = noisy_data.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  
    noisy_data = noisy_data / row_sums
    noisy_data = np.around(noisy_data, decimals=5)
    for i in range(noisy_data.shape[0]):
        row_sum = np.sum(noisy_data[i])
        if not np.isclose(row_sum, 1.0, atol=1e-5):
            print(f"Row {i} has an abnormal sum: {row_sum:.6f}")
          
    # Build DataFrame
    noisy_df = pd.DataFrame(noisy_data, columns=datas.columns)
    resampled_data = pd.concat([datas_reset, noisy_df], ignore_index=True)
    resampled_labels = pd.concat([labels_reset, minority_labels], ignore_index=True)
  
    # Construct the augmented DataFrame
    resampled_df = resampled_data.copy()
    resampled_df['Dmax'] = resampled_labels
    resampled_df.reset_index(drop=True, inplace=True)
    resampled_df['index'] = range(1, len(resampled_df) + 1)
    resampled_df.to_csv(output_path, index=False)
  
    print(f"Augmented dataset size: {len(resampled_df)}")
    print(f"Number of added samples: {len(resampled_df) - len(datas)}")
    print(f"Augmented data saved to: {output_path}")
  
    return resampled_data, resampled_labels
  
# Add Gaussian noise
def add_random_noise(row, num_features, noise_level):
    num_selected = np.random.randint(1, 9)
    selected_cols = np.random.choice(row.shape[0], size=num_selected, replace=False)
    noise = np.random.exponential(scale=noise_level, size=num_selected)
    noise_vector = np.zeros_like(row)
    noise_vector[selected_cols] = noise
    return row + noise_vector  
  
if __name__ == '__main__':
    data_path = 'train_data.csv'
    output_path = 'processed_data/processed_GN.csv'
    add_gaussian_noise(data_path, output_path)

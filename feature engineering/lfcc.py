# %%
import os
import librosa
import pandas as pd

# %%
def extract_lfcc(folder_path):
    lfcc_features = []
    for file in os.listdir(folder_path):
        if file.endswith('.wav'):
            file_path = os.path.join(folder_path, file)
            y, sr = librosa.load(file_path)

            lfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=10)
            lfcc_mean = lfcc.mean(axis=1)

            lfcc_features.append(lfcc_mean)

    return lfcc_features


# %%

folder_path = 'folder_name'
extracted_lfcc= extract_lfcc(folder_path)

# Create DataFrame with LFCC features
df = pd.DataFrame(extracted_lfcc)

# Load only the required column(target) from the metadata CSV file
metadata_df = pd.read_csv('metadata.csv', usecols=['Classname'])

# Merge DataFrames based on their indices
merged_df = pd.concat([df, metadata_df], axis=1)

# Save the updated dataframe to a new CSV file
merged_df.to_csv('lfcc_features_foldername.csv', index=False)




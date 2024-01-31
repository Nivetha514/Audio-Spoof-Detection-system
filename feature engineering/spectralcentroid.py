# %%
import os
import librosa
import pandas as pd

# %%
def extract_spectralcentroid(folder_path):
    spectral_centroid_values=[]
    for file in os.listdir(folder_path):
        if file.endswith('.flac'):
            file_path = os.path.join(folder_path, file)
            y, sr = librosa.load(file_path)

            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)

            mean_spectral_centroid = spectral_centroid.mean()

            spectral_centroid_values.append(mean_spectral_centroid)

    return spectral_centroid_values


# %%

folder_path = 'folder_name'
extracted_spectralcentroid= extract_spectralcentroid(folder_path)

# Create DataFrame with spectralcentroid features
df = pd.DataFrame(extracted_spectralcentroid)

# Load only the required column(target) from the metadata CSV file
metadata_df = pd.read_csv('metadata.csv', usecols=['Classname'])

# Merge DataFrames based on their indices
merged_df = pd.concat([df, metadata_df], axis=1)

# Save the updated dataframe to a new CSV file
merged_df.to_csv('spectralcentroid_features_foldername.csv', index=False)




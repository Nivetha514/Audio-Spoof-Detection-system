# %%
import os
import librosa
import pandas as pd

# %%
def extract_spectralrolloff(folder_path):
    spectral_rolloff_values=[]
    for file in os.listdir(folder_path):
        if file.endswith('.wav'):
            file_path = os.path.join(folder_path, file)
            y, sr = librosa.load(file_path)

            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            mean_spectral_rolloff = spectral_rolloff.mean()

            spectral_rolloff_values.append(mean_spectral_rolloff)
    return spectral_rolloff_values


# %%

folder_path = 'folder_name'
extracted_spectralrolloff= extract_spectralrolloff(folder_path)

# Create DataFrame with spectralrolloff features
df = pd.DataFrame(extracted_spectralrolloff)

# Load only the required column(target) from the metadata CSV file
metadata_df = pd.read_csv('metadata.csv', usecols=['Classname'])

# Merge DataFrames based on their indices
merged_df = pd.concat([df, metadata_df], axis=1)

# Save the updated dataframe to a new CSV file
merged_df.to_csv('spectralrolloff_features_foldername.csv', index=False)




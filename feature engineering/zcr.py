# %%
import librosa
import os
import pandas as pd

# %%
def extract_zcr(folder_path):
    zcr_values=[]
    for filename in os.listdir(folder_path):
        if filename.endswith(".wav"):
            file_path = os.path.join(folder_path, filename)
            audio,sr = librosa.load(file_path, sr=None)
            zcr=librosa.feature.zero_crossing_rate(y=audio)
            mean_zcr=zcr.mean()
            zcr_values.append(mean_zcr)
    return zcr_values


# %%

folder_path = 'folder_name'
extracted_zcr= extract_zcr(folder_path)

# Create DataFrame with zcr features
df = pd.DataFrame(extracted_zcr)

# Load only the required column(target) from the metadata CSV file
metadata_df = pd.read_csv('metadata.csv', usecols=['Classname'])

# Merge DataFrames based on their indices
merged_df = pd.concat([df, metadata_df], axis=1)

# Save the updated dataframe to a new CSV file
merged_df.to_csv('zcr_features_foldername.csv', index=False)




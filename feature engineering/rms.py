# %%
import os
import librosa
import pandas as pd

# %%
def extract_rms(folder_path):
    rms_values=[]
    for file in os.listdir(folder_path):
        if file.endswith('.wav'):
          file_path=os.path.join(folder_path, file)
          y, sr= librosa.load(file_path)

          rms=librosa.feature.rms(y=y)
          rms_mean=rms.mean()

          rms_values.append(rms_mean)

    return rms_values

# %%

folder_path = 'folder_name'
extracted_rms= extract_rms(folder_path)

# Create DataFrame with rms features
df = pd.DataFrame(extracted_rms)

# Load only the required column(target) from the metadata CSV file
metadata_df = pd.read_csv('metadata.csv', usecols=['Classname'])

# Merge DataFrames based on their indices
merged_df = pd.concat([df, metadata_df], axis=1)

# Save the updated dataframe to a new CSV file
merged_df.to_csv('rms_features_foldername.csv', index=False)




# %%
import librosa
import os
import pandas as pd

# %%
def extract_rfcc(folder_path):
    rfcc_features=[]
    for file in os.listdir(folder_path):
        if file.endswith('.wav'):
            file_path = os.path.join(folder_path, file)
            y, sr= librosa.load(file_path)

            rfcc = librosa.feature.poly_features(y=y, order=15)
            rfcc_mean=rfcc.mean(axis=1)
            rfcc_features.append(rfcc_mean)
    return rfcc_features


# %%

folder_path = 'folder_name'
extracted_rfcc= extract_rfcc(folder_path)

# Create DataFrame with RFCC features
df = pd.DataFrame(extracted_rfcc)

# Load only the required column(target) from the metadata CSV file
metadata_df = pd.read_csv('metadata.csv', usecols=['Classname'])

# Merge DataFrames based on their indices
merged_df = pd.concat([df, metadata_df], axis=1)

# Save the updated dataframe to a new CSV file
merged_df.to_csv('rfcc_features_foldername.csv', index=False)




# %% [markdown]
# MFCC FEATURES

# %%
#imports
import os
import librosa
import librosa.display
import pandas as pd

# %%
# Define the feature extraction function
def extract_mfcc(folder_path):
   mfcc_features=[]
   for file in os.listdir(folder_path):
      if file.endswith('.wav'):
         file_path = os.path.join(folder_path, file)
         y, sr = librosa.load(file_path)
         mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
         mfcc_mean=mfcc.mean(axis=1)
         mfcc_features.append(mfcc_mean)
   return mfcc_features


# %%

folder_path = 'folder_name'
extracted_mfcc= extract_mfcc(folder_path)

# Create DataFrame with MFCC features
df = pd.DataFrame(extracted_mfcc)

# Load only the required column(target) from the metadata CSV file
metadata_df = pd.read_csv('metadata.csv', usecols=['Classname'])

# Merge DataFrames based on their indices
merged_df = pd.concat([df, metadata_df], axis=1)

# Save the updated dataframe to a new CSV file
merged_df.to_csv('mfcc_features_foldername.csv', index=False)




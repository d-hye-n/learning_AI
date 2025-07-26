import os
import pandas as pd
import requests
from tqdm import tqdm

# Create a directory to store the images
if not os.path.exists('flower_images'):
    os.makedirs('flower_images')

# Read the CSV file
df = pd.read_csv('flowers.csv', header=None, names=['url', 'label'])

# Function to convert GCS URI to public URL
def gcs_to_http(gcs_uri):
    return gcs_uri.replace('gs://', 'https://storage.googleapis.com/')

# Function to download and save images
def download_image(url, label, filename):
    try:
        # Create a directory for the label if it doesn't exist
        if not os.path.exists(f'flower_images/{label}'):
            os.makedirs(f'flower_images/{label}')

        # Get the image from the URL
        response = requests.get(url, stream=True)
        response.raise_for_status()

        # Save the image
        with open(f'flower_images/{label}/{filename}', 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"Could not download {url}. Error: {e}")
        return False

# Download the images
for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    label = row['label']
    gcs_uri = row['url']
    http_url = gcs_to_http(gcs_uri)
    filename = gcs_uri.split('/')[-1]
    download_image(http_url, label, filename)

print("Image download complete.")
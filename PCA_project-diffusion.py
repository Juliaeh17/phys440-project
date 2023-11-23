import json
from google.oauth2 import service_account
from googleapiclient.discovery import build
import io
from googleapiclient.http import MediaIoBaseDownload
from PIL import Image
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Give the script access to the Google Drive which holds the X-ray images of the pneumonia patient's lungs

# Keys to the Google service account
service_account_info = {
  "type": "service_account",
  "project_id": "cnn-pneumonia",
  "private_key_id": "6e854c3912895eb5814354b4c53c1045cfbc558e",
  "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQCabF3eegPqZxe2\nw3BoGFFub+attPEKSBsRqdCIWXZQmHJNeMNdlzAdU6hKONW45Cye1lL4YDAbMlTV\nWBM6hkU4vitQKwKS9JOP8feVUXwv619VWmQLiuEQ7WRBxtT/Jv6nZSvc209x8I4i\nydfQDbOaV7sVP0sPoJ2A7+Wtz7KUWuC/Wewy+J9HVeGaT8YvuczBB/awoP6fLVEt\nUxwcAfswkhrWC2wuvnDMcVj5kcqVUj7YqCF9LfUCCZQvPCm5KuK3MRpblQCQTo9z\nMRaFKObn5hplBcubulDXBiNpC5Ibn3ziuWZq0m2GyR0/aSjkowUZAv9z2CEnwRLD\ntm7LjKm/AgMBAAECggEAIT4KxwXFKgSfWjygSghSH0//fI1jBi+XhCnmNjPsAFWQ\n5ZSFmfQd63JC3Bd4CqEz6c2Bdu5d7Lzc/kBDg2m8JQbrFAruNuxnh6ky/vXXogkt\nMJaQyttOr/Iqju5ak8K1NxvUYWrko0aBkoOY7bTFYQhZwa0qX2bYARWjf9MiQkUo\nR7yVdgmuPSeNGovhtaszgghNCII9Q4dHudCGt5v7LCU8j114JZh5PudPS0ygZOxt\n7zxhpx2Z/6T2mveMNXeytcAVd4pI1kHNtRssFQQXRsHQHa8njgeejzGIZ85EitmQ\nvnFmFZcwz5TzM2JlMYXPSSFQ87F1G6v80Es8lk9tYQKBgQDPHdid/GDN9wYXEwMg\nWHBg1XNzk22ev75gmC3AEWlTs2uyxp1x9YijPDQTYSTktadHLjyECLlzDrPD+TjR\nKfItwyuISL0Y084ULfKuovd9RKQMD4OQxgItQehWJwt4BQzkzby2J+ttGHJsn09B\nWqniRrRjDFiAijg9x+v9/T+VtQKBgQC+3r/Zq4oqwcyFLHrzegBPJ66osoRYSsYN\nhfi2HUYTtS03azXOnjv+Wbgv/a4VPgK9rZJuSP3QdnBWfZ0E2li2c3/QgHK3v8rz\nxya44fjdgEDXsC2TXVG2uskIPlKfVIUscPu8QYAJFzUIlv/WMXCqWmOfSg8JZ5Xj\nWZEoH02qIwKBgB/aSDkr2jty1SXxT5bG4ymRSjspHj++32l9nfOe+eLcgiCxeP9Q\nsp3gIWYll4XxBfPlgXsQ2GyAg1cNWhaY69zr2iMQhLxvvo7N7je7anKCfvQ34pT+\nTtFlBHVTdekUZcI+fdpJ02Qo0VgxPAAiEGRzWUSuXmIOX83olDuwmfnNAoGBAL4N\nPiqXS/RXFDZmm+ZjzHsEoD0JxA3GJn7Ar21mqKhm9qb/8YvSsxoIbAYdKoGsRT5o\n3i1CMLiptiHo4bg7UaoaR2JtA81DA+rImh3to8eqNOaPXlIl0X+JbTLwG4Tau+AM\ngBxre/mRShVLhWLZx71YU2oAbAiJRA4k2QyCXurRAoGABBIm6Oi7KAGPRZdBEoFG\nb4EC5XTEDMdkhwEfvrVMPOimpp54WQ5QPHdQtTr96Lh8ZvEYCLppNMnP0UboLQFx\nzM7x3lfxrCtKiPkyglT1mt92XMRe+ZD0bjWhBDFgQ+2uwvRXHZ3S7CmgJla7O+kX\nspMnW9EErNtzpbfcv7QQweI=\n-----END PRIVATE KEY-----\n",
  "client_email": "images@cnn-pneumonia.iam.gserviceaccount.com",
  "client_id": "104602856143137945445",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/images%40cnn-pneumonia.iam.gserviceaccount.com",
  "universe_domain": "googleapis.com"
}

# Level of access script is given via the service account
CREDENTIALS = service_account.Credentials.from_service_account_info(service_account_info)

# Build access to the Google Drive as argument to be used in code
service = build('drive', 'v3', credentials=CREDENTIALS)

# load images to code
def get_images(service, folder_id_layer1, folder_id_layer2):
    """
    Retrieves images from two nested folder layers in Google Drive and 
    compiles them into a single numpy array.
    All images found in the folders specified by 'folder_id_layer1' and 'folder_id_layer2' 
    are loaded and combined into a 2D numpy array.
    """
    images = []

    # Function to list image files in a given folder
    def list_image_files(service, folder_id):
        query = f"'{folder_id}' in parents and mimeType contains 'image/'"
        response = service.files().list(q=query).execute()
        return response.get('files', [])

    # Function to download an image
    def download_image(service, file_id):
        request = service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
        fh.seek(0)
        return Image.open(fh)

    # Process the second layer folder
    image_files_layer2 = list_image_files(service, folder_id_layer2)
    for image_file in image_files_layer2:
        image = download_image(service, image_file['id'])
        image_array = np.array(image).flatten()
        images.append(image_array)

    # Combine all images into a single numpy array
    return np.vstack(images)

processed_train = '1o54sqskHTEE0DahHfVFIKTHIzjHEzC5v'
processed_test = '1Bz5TwZAXQzrdBFnd5oIXQ1vRpZ4fkRW_'
NORMAL_train = '1XEFLMZZrTgrLr58ByJgiHWCZCvrfVPIG'
PNEUMONIA_train = '14D1l5yJ581WpbrrgbZpgcP72uukPm2r3'
NORMAL_test = '12THDO48S0Z0-84TkRtQ_HX--gtWEERqH'
PNEUMONIA_test = '1a16E33tNfVHojsR2mUApYQO2uUxi-UNH'
images_1 = get_images(service, processed_train, NORMAL_train)
images_2 = get_images(service, processed_train, PNEUMONIA_train)
images_3 = get_images(service, processed_test, NORMAL_test)
images_4 = get_images(service, processed_test, PNEUMONIA_test)

all_images = np.vstack((images_1, images_2, images_3, images_4))
all_images.shape

# 2D PCA
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(all_images)

plt.figure(figsize=(10, 6))
plt.scatter(reduced_data[:, 0], reduced_data[:, 1])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Images')
plt.savefig('PCA plot.png')
plt.show()

# Number of clusters
n_clusters = 2  # adjust based on your needs

# Apply K-Means clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
labels = kmeans.fit_predict(reduced_data)

plt.figure(figsize=(10, 6))
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Images with K-Means Clustering')
plt.show()


# 3D PCA
pca = PCA(n_components=3)
reduced_data_3 = pca.fit_transform(all_images)

# Plot 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Assuming 'reduced_data' is your data array with shape (n_samples, 3)
ax.scatter(reduced_data_3[:, 0], reduced_data_3[:, 1], reduced_data_3[:, 2])

ax.set_xlabel('Component 1')
ax.set_ylabel('Component 2')
ax.set_zlabel('Component 3')
plt.title('3D PCA Plot')

plt.show()

# Kmeans of 3d plot
n_clusters_3d = 3  # Number of clusters

kmeans = KMeans(n_clusters=n_clusters_3d, random_state=0)
labels_3d = kmeans.fit_predict(reduced_data_3)

# plot with Kmeans
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot of your data with the color representing the cluster label
ax.scatter(reduced_data_3[:, 0], reduced_data_3[:, 1], reduced_data_3[:, 2], c=labels_3d, cmap='viridis')
ax.set_xlabel('Dimension 1')
ax.set_ylabel('Dimension 2')
ax.set_zlabel('Dimension 3')
plt.title('3D K-Means Clustering')

plt.show()

import plotly.graph_objects as go
from plotly.offline import init_notebook_mode
init_notebook_mode(connected=True)

fig = go.Figure(data=[go.Scatter3d(
    x=reduced_data_3[:, 0],  # PC1
    y=reduced_data_3[:, 1],  # PC2
    z=reduced_data_3[:, 2],  # PC3
    mode='markers',
    marker=dict(
        size=5,
        color=labels_3d,  # Assign cluster labels to colors
        colorscale='Viridis',  # Define the colorscale
        opacity=0.8
    )
)])

# Customize the layout
fig.update_layout(
    title='3D PCA Plot with K-Means Clustering',
    scene=dict(
        xaxis_title='Principal Component 1',
        yaxis_title='Principal Component 2',
        zaxis_title='Principal Component 3'
    ),
    margin=dict(l=0, r=0, b=0, t=0)  # Tight layout
)

fig.show('browser')


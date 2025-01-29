import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
import torch

file_path = 'logging/cifar100_test/tta_grad/fedtsa_original_feature_adapt_fedpl_client_niid_resnext29_lp_1_2025_01_27_10_50_36/collaboration.pkl'
with open(file_path, 'rb') as f:
    collaboration_graph = pickle.load(f)

# Stack the collaboration graphs into a single tensor
collaboration_graph = torch.stack(collaboration_graph).cpu()

# Number of frames in the animation
num_frames = collaboration_graph.shape[0]

for i in range(50):
    plt.figure(figsize=(8, 6))
    sns.heatmap(collaboration_graph[i], annot=False, cmap="viridis", cbar=True) # Set cbar to True to show the colorbar
    plt.title(f"{i+1}/750")
    plt.savefig(f'plots_tsa/{i+1}.png')
    plt.close()

from PIL import Image
import os

# Directory containing images
image_dir = "plots_tsa"
output_gif = "collaboration_graph_heatmap_fedtsa.gif"



# Collect all image file paths in the directory
image_files = [os.path.join(image_dir, f'{i+1}.png') for i in range(50)]
# Open the images and create a GIF
images = [Image.open(img) for img in image_files]
image_files = image_files[:50]

images[0].save(
    output_gif,
    save_all=True,
    append_images=images[1:],
    duration=1000,  # Duration between frames in milliseconds
    loop=0  # Loop forever (set to 1 for a single loop)
)

print(f"GIF saved as {output_gif}")
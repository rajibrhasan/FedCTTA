
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import torch
import os

# os.makedirs('cgraph')


def save_graph(fg, tsa, ours, num):
    vmin = min(fg.min(), tsa.min(), ours.min())
    vmax = max(fg.max(), tsa.max(), ours.max())

    # Create figure and axes_
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    sns.heatmap(fg, ax=axes[0], cmap="viridis", vmin=vmin, vmax=vmax, cbar=False)
    sns.heatmap(tsa, ax=axes[1], cmap="viridis", vmin=vmin, vmax=vmax, cbar=False)
    im = sns.heatmap(ours, ax=axes[2], cmap="viridis", vmin=vmin, vmax=vmax, cbar=False)

    # # Plot heatmaps (without colorbar)
    # sns.heatmap(fg, ax=axes[0], cmap="viridis", vmin=vmin, vmax=vmax, cbar=False)
    # sns.heatmap(tsa, ax=axes[1], cmap="viridis", vmin=vmin, vmax=vmax, cbar=False)
    # sns.heatmap(ours, ax=axes[2], cmap="viridis", vmin=vmin, vmax=vmax, cbar=False)

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.1, 0.02, 0.8])
    fig.colorbar(im.collections[0], cax=cbar_ax)

    # Add shared colorbar
    # cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    # sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=vmin, vmax=vmax))
    # fig.colorbar(sm, cax=cbar_ax)

    # Save the figure
    plt.savefig(f"cgraph/heatmaps_{num}.png", dpi=300, bbox_inches="tight")
    # plt.savefig("heatmaps_shared_colorbar.pdf", bbox_inches="tight")
    plt.show()



fedgraph = 'logging/cifar10_test/tta_adapt/niid/fedgraph_original_feature_adapt_fedpl_client_lp_1_seed456_1738086233/collaboration.pkl'
fedtsa = 'logging/cifar10_test/tta_adapt/niid/fedtsa_original_feature_adapt_fedpl_client_lp_1_seed3_1738008932/collaboration.pkl'
ours = 'logging/cifar10_test/tta_adapt/niid/ours_random_output_adapt_fedpl_client_lp_1_seed3_1738014263/collaboration.pkl'

with open(fedgraph, 'rb') as f:
    pgraph = pickle.load(f)
    pgraph = torch.stack(pgraph).cpu()

with open(fedtsa, 'rb') as f:
    tsa = pickle.load(f)
    tsa = torch.stack(tsa).cpu()

with open(ours, 'rb') as f:
    ours = pickle.load(f)
    ours = torch.stack(ours).cpu()



save_graph(pgraph[0], tsa[0], ours[0], 0)
save_graph(pgraph[374], tsa[374], ours[374], 374)
save_graph(pgraph[749], tsa[749], ours[749], 749)





# plt.figure(figsize=(8, 6))
# sns.heatmap(collaboration_graph[-1], annot=False, cmap="viridis", cbar=True) # Set cbar to True to show the colorbar
# plt.title("Collaborative Matrix Heatmap")
# plt.savefig('collab_mat_original.png')
# plt.show()
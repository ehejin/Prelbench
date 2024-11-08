import numpy as np
import umap.umap_ as umap
import matplotlib.pyplot as plt

# Load embeddings
embeddings = np.load('./embeddings5.npy') 
plt.figure(figsize=(10, 5))

#embeddings = embeddings[:800]
print(embeddings.shape)

umap_reducer = umap.UMAP(n_components=2, random_state=42)
umap_embeddings = umap_reducer.fit_transform(embeddings)

plt.subplot(1, 2, 2)
plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], s=5, alpha=0.7, c='orange')
plt.title('UMAP Embeddings')
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')

plt.tight_layout()
output_path = "./umap_embeddings_plot.png"
plt.savefig(output_path)
plt.show()

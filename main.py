import torch
from hypergraph import Hypergraph
from transformer_encoder import TransformerEncoder
from vae import VariationalAutoEncoder
from utils import load_data, retrieve_top_k_labels

# Hyperparameters
LAMBDA = 0.6
TEMPERATURE = 0.07
TOP_K = 5

def main():
    # Load documents and knowledge base (UMLS)
    documents, umls_data = load_data()

    # Initialize Hypergraph, Transformer Encoder, and VAE
    hypergraph = Hypergraph(umls_data)
    encoder = TransformerEncoder()
    vae = VariationalAutoEncoder()

    # Process each document
    for doc in documents:
        # Construct hypergraph for the document
        G = hypergraph.construct(doc)

        # Encode nodes and hyperedges
        z_v, z_e = encoder(G)

        # Generate augmented hypergraph views
        G_aug, recon_loss = vae.generate_augmented_graph(G, z_v, z_e)

        # Compute contrastive loss
        contrastive_loss = encoder.compute_contrastive_loss(z_v, z_e, TEMPERATURE)

        # Combine losses
        total_loss = contrastive_loss + LAMBDA * recon_loss

        print(f"Document: {doc['id']}, Total Loss: {total_loss.item()}")

    # Perform zero-shot label retrieval
    for doc in documents:
        predicted_labels = retrieve_top_k_labels(doc, encoder, TOP_K)
        print(f"Document: {doc['id']}, Predicted Labels: {predicted_labels}")

if __name__ == "__main__":
    main()

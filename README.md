🧠 Violence Behavior Analysis Dashboard
PCA + kNN Similarity Explorer with GNN-Inspired Architecture (Streamlit App)

This project builds an interactive analytical dashboard to study behavioral patterns of violent offenders (mass shooters dataset).
The dashboard allows users to:

Explore the dataset in a visual, intuitive way

Understand behavioral clusters using Principal Component Analysis (PCA)

Compute and visualize nearest behavioral neighbors (kNN)

Predict shooter categories & view similarity connections

Explore an interactive kNN Graph Network (PyVis)

Although the deployment version uses KNN instead of a full GNN, the architecture and modeling pipeline are inspired by Graph Neural Networks (GNNs) such as GraphSAGE, which was used during experimentation.

🔍 Project Motivation

Mass violence incidents often involve multiple interacting behavioral, social, psychological, and situational factors.
Traditional classification models struggle because:

✔ The dataset is high-dimensional
✔ Many features are correlated
✔ Behavior patterns resemble a graph, not isolated data points

This project attempts to bridge:

Behavioral psychology + Machine learning + Graph representation learning
🚀 Key Features
🔹 1. PCA Behavioral Component Analysis

We extracted 32 PCA components summarizing:

Social Contagion

Signs of Crisis

Offender Background

Trauma History

Weapon Interest

Mental Health Indicators

Motivation Factors

Crime & Violence History

Each PCA axis is human-interpretable (e.g., “Planning + Social Media Use”).

The Streamlit UI displays them with real names instead of PCA 1, 2, 3.

🔹 2. k-Nearest Neighbor Similarity (kNN)

For each offender, we compute:

Top behavioral neighbors

Distance in PCA space

Comparison of fatalities

Behavior similarity clusters

This answers questions like:

“Which previous offenders did this shooter most resemble behaviorally?”

🔹 3. Prediction Interface

Two modes:

⭐ Select Existing Shooter

Predict weapon type for a real entry.

⭐ Custom Behavioral Input

Manually adjust PCA-level behavioral traits and generate predictions.

🔹 4. Interactive Graph Explorer

We create a kNN graph (network):

Nodes = offenders

Edges = “behaviorally similar” connections

Visualized using PyVis

Fully interactive (hover, drag, zoom)

🔹 5. GNN Motivation (Research Phase)

Originally, the project used:

GraphSAGE (PyTorch Geometric)

Learned embeddings from graph structure

Combined PCA + GNN classifier

However:

✓ Streamlit Cloud does not support PyTorch / PyTorch Geometric
✓ Final deployment uses lightweight, accessible kNN + PCA backend

The README reflects this evolution honestly.

🛠️ Tech Stack
Component	Technology
Dashboard	Streamlit
Modeling	PCA, KNN
Graph Visualization	PyVis
Data Processing	Pandas, NumPy
Experimentation (local)	GraphSAGE, PyTorch Geometric
Deployment	Streamlit Cloud
📁 Repository Structure
📦 violence-gnn-app
 ┣ 📄 app.py                # Main Streamlit application
 ┣ 📄 requirements.txt      # Minimal dependencies (no PyTorch needed)
 ┣ 📄 cleaned_features.csv  # Dataset used for predictions
 ┣ 📄 pca.npy               # PCA-transformed feature matrix
 ┣ 📄 pca_model.pkl         # PCA model (optional)
 ┣ 📄 README.md             # Project documentation
 ┗ 📄 .streamlit/config.toml (optional)

📊 How the Model Works (Simplified)
1️⃣ PCA reduces 100+ behavioral features → 32 interpretable dimensions
2️⃣ kNN finds similar offenders in PCA space
3️⃣ Simple classifier predicts weapon category
4️⃣ Graph Explorer creates links between similar offenders

This results in a behavior-driven similarity graph, not just a classifier.

🎯 Use Cases

✔ Criminal psychology research
✔ Behavioral modeling
✔ Threat assessment and early warning indicators
✔ Academic demonstration of PCA + graphs
✔ GNN conceptual learning through a real dataset

🧪 Future Improvements

Re-enable GraphSAGE once GPU-compatible deployment is available

Integrate SHAP explainability

Cluster offenders using DBSCAN or HDBSCAN

Build an early-warning risk scoring system

Add model training notebook inside repo

🙌 Acknowledgements

This project is inspired by:

Research on mass violence behavioral analysis

GNN literature (GraphSAGE, GAT, GCN)

Practical deployment constraints on Streamlit Cloud

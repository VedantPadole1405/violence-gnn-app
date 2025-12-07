import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from pyvis.network import Network
import streamlit.components.v1 as components
import joblib
import plotly.express as px


# ==============================================
# PCA NAMES (unchanged)
# ==============================================
pca_names = [
    "PCA 1 — Planning, Past-Violence Interest, Performance",
    "PCA 2 — Crisis Timeline, Losing Touch, Crisis Signs",
    "PCA 3 — Education, Employment Type, School Performance",
    "PCA 4 — Relationship Status, Children, Social Media Use",
    "PCA 5 — Employment Motive, Insider/Outsider, Workplace Shooting",
    "PCA 6 — Military Branch, Siblings Count, Older Siblings",
    "PCA 7 — Victims Injured, Firearms Brought, Race",
    "PCA 8 — Violent Video Games, Year, Social Media",
    "PCA 9 — Isolation, Parental Suicide, Leakage",
    "PCA 10 — Depressed Mood, Terror Group, Physical Altercations",
    "PCA 11 — Pop Culture Link, Prior Hospitalization, Involuntary Hospitalization",
    "PCA 12 — Parental Death, Single Parent, Depressed Mood",
    "PCA 13 — Other Location, Multiple Locations, Sexual Orientation",
    "PCA 14 — Siblings Count, Younger Siblings, Immigrant",
    "PCA 15 — Firearm Interest, Recent Treatment, Known to Police",
    "PCA 16 — Known to Police, Homophobia Motive, Health Issues",
    "PCA 17 — Sexual Abuse, Bullying, Urbanicity",
    "PCA 18 — Leakage, Parental Divorce, Legal Motive",
    "PCA 19 — Head Injury/TBI, Animal Abuse, Calm/Happy",
    "PCA 20 — Terror Group, Month, Racism/Xenophobia",
    "PCA 21 — Partner Victim, Relationship Motive, Health Issues",
    "PCA 22 — Sexual Orientation, Day, Autism",
    "PCA 23 — Sexual Offenses, Interpersonal Conflict, Year",
    "PCA 24 — Racism Motive, Partner Victim, Case Outcome",
    "PCA 25 — Employment Type, Employment Status, Legal Motive",
    "PCA 26 — Gang Affiliation, Animal Abuse, Location Details",
    "PCA 27 — Pop Culture, Planning, Depressed Mood",
    "PCA 28 — State Code, Parental Divorce, Region",
    "PCA 29 — Gender, Known to Police, Employment Status",
    "PCA 30 — Physical Altercations, Gender, Conflict Motive",
    "PCA 31 — Past Violence Interest, Community Involvement, Abusive Behavior",
    "PCA 32 — Parental Suicide, Leakage, Education"
]


# ==============================================
# LOAD DATA
# ==============================================
@st.cache_resource
def load_data():
    df = pd.read_csv("cleaned_features.csv")
    X_pca = np.load("pca.npy")

    try:
        pca_model = joblib.load("pca_model.pkl")
    except:
        pca_model = None

    return df, X_pca, pca_model


df, X_pca, pca_model = load_data()

possible_labels = ["weapon_type", "classification", "class", "y"]
label_col = next((c for c in possible_labels if c in df.columns), df.columns[-1])

possible_victims = ["Number_Killed", "People_Killed", "Victims_Killed", "Fatalities"]
victims_col = next((c for c in possible_victims if c in df.columns), None)


# ==============================================
# KNN MODEL (for predictions)
# ==============================================
knn = NearestNeighbors(n_neighbors=8)
knn.fit(X_pca)


# ==============================================
# STREAMLIT UI
# ==============================================
st.set_page_config(page_title="Violence Project Dashboard", layout="wide")
tabs = st.tabs(["📊 Dataset Explorer", "🔮 Prediction (kNN)", "🕸️ Graph Explorer"])


# ==============================================
# TAB 1 — Dataset Explorer
# ==============================================
with tabs[0]:
    st.header("📊 Dataset Explorer")
    st.dataframe(df.head())

    fig = px.scatter(
        x=X_pca[:, 0],
        y=X_pca[:, 1],
        color=df[label_col].astype(str),
        labels={"x": "PC1", "y": "PC2"},
        title="PCA Projection of Shooter Features"
    )
    st.plotly_chart(fig, use_container_width=True)


# ==============================================
# TAB 2 — Prediction (kNN only)
# ==============================================
with tabs[1]:

    st.header("🔮 Prediction Based on PCA + kNN")

    mode = st.radio("Choose input mode:", ["Select Shooter", "Custom PCA Input"])

    # ---------------------------------------------------------
    # OPTION 1 — EXISTING SHOOTER
    # ---------------------------------------------------------
    if mode == "Select Shooter":
        idx = st.selectbox("Choose Shooter ID:", df.index)

        if st.button("Predict Using kNN"):
            distances, neighbors = knn.kneighbors([X_pca[idx]])
            neigh_ids = neighbors[0]

            st.subheader("👥 Nearest Neighbors")
            knn_df = pd.DataFrame({
                "Shooter ID": neigh_ids,
                "Distance": distances[0],
            })

            if victims_col:
                knn_df["People Killed"] = df.loc[neigh_ids, victims_col].values

            st.table(knn_df)

            # Fatality analytics
            if victims_col:
                st.subheader("🔥 Fatality Impact Among Neighbors")
                st.write(f"Total fatalities: {knn_df['People Killed'].sum()}")
                st.write(f"Average fatalities: {knn_df['People Killed'].mean():.2f}")
                st.write(f"Max fatalities: {knn_df['People Killed'].max()}")


    # ---------------------------------------------------------
    # OPTION 2 — CUSTOM PCA INPUT
    # ---------------------------------------------------------
    else:
        st.subheader("Enter PCA Behavioral Features")
        mean_vals = X_pca.mean(axis=0)
        custom_vals = []

        for i in range(32):
            custom_vals.append(st.number_input(pca_names[i], value=float(mean_vals[i])))

        if st.button("Predict for Custom Input"):
            x = np.array(custom_vals).reshape(1, -1)

            distances, neighbors = knn.kneighbors(x)
            neigh_ids = neighbors[0]

            st.subheader("👥 Nearest Neighbors")
            knn_df = pd.DataFrame({"Shooter ID": neigh_ids})

            if victims_col:
                knn_df["People Killed"] = df.loc[neigh_ids, victims_col].values

            st.table(knn_df)

            if victims_col:
                st.subheader("🔥 Fatality Impact Among Neighbors")
                st.write(f"Total fatalities: {knn_df['People Killed'].sum()}")
                st.write(f"Average fatalities: {knn_df['People Killed'].mean():.2f}")
                st.write(f"Max fatalities: {knn_df['People Killed'].max()}")


# ==============================================
# TAB 3 — Graph Explorer
# ==============================================
with tabs[2]:
    st.header("🕸️ kNN Graph Explorer")

    g = Network(height="700px", width="100%", bgcolor="#FFFFFF", directed=False)
    distances, neighbors = knn.kneighbors(X_pca)

    for i in range(len(df)):
        g.add_node(int(i), label=str(i))

    for i in range(len(df)):
        for nb in neighbors[i]:
            g.add_edge(int(i), int(nb))

    g.save_graph("graph.html")
    components.html(open("graph.html", "r").read(), height=700)

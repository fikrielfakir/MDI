"""
Deep Learning from Scratch - Interactive Streamlit Application
A comprehensive educational tool for understanding neural networks.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import io
from activations import ActivationFunctions, visualize_activations
from perceptron import Perceptron, demo_logic_gates, demonstrate_xor_problem
from mlp import MLP, solve_xor_with_mlp
from data_utils import (load_iris_dataset, load_mnist_dataset, get_mnist_sample_images,
                        compute_metrics, format_confusion_matrix, one_hot_encode)
from db_utils import (init_database, import_iris_from_csv, get_iris_sample_count,
                      save_training_run, get_training_history, get_training_run_details,
                      delete_training_run, save_trained_model, load_trained_model, get_saved_model_names,
                      save_iris_reference_image, get_iris_reference_images, get_reference_image_count,
                      delete_iris_reference_image, save_iris_image_features, get_all_iris_image_features,
                      get_feature_count, save_iris_image_model, load_iris_image_model)
from image_utils import (extract_features, normalize_features, batch_extract_features,
                         load_image_from_bytes, SPECIES_MAPPING, SPECIES_ID_MAPPING, get_feature_dimension)
import os
from sklearn.preprocessing import StandardScaler
from PIL import Image

st.set_page_config(
    page_title="Apprentissage Profond de Zéro",
    page_icon="🧠",
    layout="wide"
)

def main():
    st.title("Apprentissage Profond de Zéro")
    st.markdown("### Un Guide Interactif sur les Réseaux de Neurones")
    
    init_database()
    
    tabs = st.tabs([
        "Introduction aux RNA",
        "Fonctions d'Activation", 
        "Perceptron",
        "Perceptron Multicouche",
        "Prédiction Iris",
        "Entraînement sur Iris",
        "Entraînement sur MNIST",
        "Historique d'Entraînement"
    ])
    
    with tabs[0]:
        show_introduction()
    
    with tabs[1]:
        show_activation_functions()
    
    with tabs[2]:
        show_perceptron()
    
    with tabs[3]:
        show_mlp()
    
    with tabs[4]:
        show_iris_combined()
    
    with tabs[5]:
        show_iris_training()
    
    with tabs[6]:
        show_mnist_training()
    
    with tabs[7]:
        show_training_history()


def show_introduction():
    st.header("Introduction aux Réseaux de Neurones Artificiels")
    
    st.markdown("""
    ## Qu'est-ce qu'un Réseau de Neurones Artificiel ?
    
    Un **Réseau de Neurones Artificiel (RNA)** est un modèle informatique inspiré de la structure 
    et du fonctionnement des réseaux neuronaux biologiques dans le cerveau. Tout comme notre cerveau 
    est composé de milliards de neurones interconnectés qui traitent l'information, les RNA sont 
    constitués de neurones artificiels (également appelés nœuds ou unités) organisés en couches.
    """)
    
    st.markdown("---")
    
    st.subheader("Les Éléments Fondamentaux des Réseaux de Neurones")
    
    concept_tabs = st.tabs(["Neurones", "Poids et Biais", "Fonctions d'Activation", "Fonctions de Perte", "Optimiseurs"])
    
    with concept_tabs[0]:
        st.markdown("""
        ## Neurones - Le Cœur des Réseaux de Neurones
        
        Les neurones sont au cœur de tout réseau de neurones, y compris le perceptron. Ces entités numériques 
        reçoivent des entrées, appliquent des poids et produisent une sortie. Les neurones sont les blocs 
        de construction à travers lesquels l'information circule dans un réseau de neurones. Tout comme 
        les neurones de notre cerveau communiquent, ces blocs fondamentaux communiquent dans le langage des nombres.
        
        ### Inspiration Biologique
        
        Dans le cerveau, un neurone :
        1. Reçoit des signaux électriques d'autres neurones via les **dendrites**
        2. Traite ces signaux dans le **corps cellulaire**
        3. Si le signal combiné dépasse un seuil, il émet une sortie via l'**axone**
        4. La sortie se connecte à d'autres neurones via les **synapses**
        
        ### Modèle du Neurone Artificiel
        
        De même, un neurone artificiel :
        1. Reçoit des entrées d'autres neurones ou des données brutes
        2. Calcule une somme pondérée de ces entrées
        3. Applique une fonction d'activation pour produire une sortie
        4. Transmet la sortie à la couche suivante
        
        ### Modèle Mathématique
        
        Pour un neurone unique avec des entrées $x_1, x_2, ..., x_n$ :
        
        $$z = \\sum_{i=1}^{n} w_i x_i + b = w_1 x_1 + w_2 x_2 + ... + w_n x_n + b$$
        
        $$a = f(z)$$
        
        Où :
        - $w_i$ sont les **poids** (forces de connexion)
        - $b$ est le **biais** (ajustement du seuil)
        - $f$ est la **fonction d'activation** (introduit la non-linéarité)
        - $a$ est l'**activation** (sortie du neurone)
        """)
        
        st.markdown("""
        ```
        Signaux d'Entrée    Poids       Traitement du Neurone   Sortie
        
            x₁ ──────────── w₁ ──┐
                                 │
            x₂ ──────────── w₂ ──┼──> [Σ + b] ──> [f(z)] ──> Sortie
                                 │
            x₃ ──────────── w₃ ──┘
        ```
        """)
    
    with concept_tabs[1]:
        st.markdown("""
        ## Poids et Biais
        
        Les poids et les biais sont les **paramètres ajustables** du réseau qui influencent 
        l'importance des caractéristiques d'entrée et établissent un seuil d'activation.
        
        ### Poids
        
        Chaque caractéristique des données d'entrée reçoit un certain **poids**, indiquant son importance dans la décision :
        
        - **Poids élevé** : L'entrée a une forte influence sur la sortie
        - **Poids faible** : L'entrée a une faible influence sur la sortie
        - **Poids négatif** : L'entrée a une relation inverse avec la sortie
        
        Pendant l'entraînement, le réseau apprend quelles caractéristiques sont les plus importantes en ajustant ces poids.
        
        ### Biais
        
        Les biais agissent comme l'**exigence minimale** pour qu'une caractéristique contribue à la sortie :
        
        - Un biais décale la fonction d'activation horizontalement
        - Il permet aux neurones de s'activer même lorsque toutes les entrées sont nulles
        - Les biais offrent de la flexibilité dans la frontière de décision
        
        ### Le Processus d'Apprentissage
        
        L'ajustement des poids et des biais pendant l'entraînement du modèle affine la capacité du réseau 
        à faire des prédictions précises. L'objectif est de trouver la combinaison optimale qui minimise 
        l'erreur de prédiction.
        
        $$\\text{Sortie} = f\\left(\\sum_{i} w_i \\cdot x_i + b\\right)$$
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Interprétation des Poids :**
            | Valeur du Poids | Signification |
            |-----------------|---------------|
            | w > 0 | Corrélation positive |
            | w < 0 | Corrélation négative |
            | w ≈ 0 | Caractéristique peu impactante |
            | |w| grand | Forte influence |
            """)
        with col2:
            st.markdown("""
            **Rôle du Biais :**
            - Agit comme un seuil
            - Permet le décalage de l'activation
            - Offre de la flexibilité
            - Indépendant de l'entrée
            """)
    
    with concept_tabs[2]:
        st.markdown("""
        ## Fonctions d'Activation
        
        Les fonctions d'activation, comme la fonction échelon dans le perceptron, déterminent 
        si le neurone **s'active**. En d'autres termes, elles décident si l'information qui circule 
        doit être autorisée à contribuer à la sortie.
        
        ### Le Concept de Seuil
        
        Imaginez-le comme un seuil : si les données entrantes dépassent un certain niveau, le perceptron 
        « s'active » ou produit une sortie ; sinon, il reste silencieux. Ce processus de décision binaire 
        illustre l'essence des fonctions d'activation dans la formation de la sortie de nos neurones numériques.
        
        ### Pourquoi la Non-linéarité est Importante
        
        Sans fonctions d'activation, un réseau de neurones ne serait qu'une **transformation linéaire** :
        
        $$y = W_2 \\cdot (W_1 \\cdot x + b_1) + b_2 = W_{combined} \\cdot x + b_{combined}$$
        
        Peu importe le nombre de couches empilées, le résultat resterait linéaire ! Les fonctions d'activation 
        introduisent la **non-linéarité**, permettant aux réseaux d'apprendre des motifs complexes.
        
        ### Fonctions d'Activation Courantes
        
        | Fonction | Formule | Plage de Sortie | Cas d'Utilisation |
        |----------|---------|-----------------|-------------------|
        | **Sigmoïde** | $\\sigma(z) = \\frac{1}{1+e^{-z}}$ | (0, 1) | Classification binaire, portes LSTM |
        | **Tanh** | $\\tanh(z) = \\frac{e^z - e^{-z}}{e^z + e^{-z}}$ | (-1, 1) | Couches cachées, RNN |
        | **ReLU** | $\\max(0, z)$ | [0, ∞) | Plupart des couches cachées |
        | **Softmax** | $\\frac{e^{z_i}}{\\sum_j e^{z_j}}$ | (0, 1), somme=1 | Sortie multi-classe |
        """)
    
    with concept_tabs[3]:
        st.markdown("""
        ## Fonctions de Perte
        
        Les fonctions de perte **quantifient la différence** entre la sortie prédite et la cible réelle. 
        L'objectif est de minimiser cette différence pendant le processus d'entraînement.
        
        ### Le Concept d'Erreur
        
        Dans le contexte des réseaux de neurones, le concept d'erreur ou de perte devient évident. Le réseau 
        apprend en **réduisant la différence** entre sa prédiction et la cible réelle, posant ainsi 
        les bases pour comprendre des fonctions de perte plus sophistiquées dans des architectures avancées.
        
        ### Fonctions de Perte Courantes
        
        | Fonction de Perte | Formule | Cas d'Utilisation |
        |-------------------|---------|-------------------|
        | **Erreur Quadratique Moyenne (MSE)** | $\\frac{1}{n}\\sum(y - \\hat{y})^2$ | Problèmes de régression |
        | **Entropie Croisée** | $-\\sum y \\log(\\hat{y})$ | Problèmes de classification |
        | **Entropie Croisée Binaire** | $-[y\\log(\\hat{y}) + (1-y)\\log(1-\\hat{y})]$ | Classification binaire |
        
        ### Comment la Perte Guide l'Apprentissage
        
        1. **Passe Avant** : Le réseau fait une prédiction
        2. **Calcul de la Perte** : Comparer la prédiction à la valeur réelle
        3. **Passe Arrière** : Calculer les gradients de la perte par rapport aux poids
        4. **Mise à Jour** : Ajuster les poids pour réduire la perte
        
        L'objectif de l'entraînement est de trouver les poids qui **minimisent la fonction de perte**.
        """)
        
        st.info("Perte faible = Meilleures prédictions. Le processus d'entraînement réduit itérativement la perte jusqu'à ce que le modèle fonctionne bien.")
    
    with concept_tabs[4]:
        st.markdown("""
        ## Optimiseurs
        
        Bien que plus simples dans le contexte du perceptron, les optimiseurs sont essentiels pour 
        **ajuster les poids et les biais** en fonction de la perte calculée. Ils affinent les paramètres 
        du modèle pour minimiser la perte et améliorer les performances globales.
        
        ### Le Rôle des Optimiseurs
        
        Ce mécanisme laisse entrevoir les techniques d'optimisation plus larges dans des architectures 
        d'apprentissage profond plus complexes. Les optimiseurs déterminent :
        
        - **À quelle vitesse** mettre à jour les poids (taux d'apprentissage)
        - **Dans quelle direction** se déplacer (direction du gradient)
        - **Combien** ajuster chaque paramètre
        
        ### Techniques d'Optimisation Populaires
        
        | Optimiseur | Description | Caractéristiques |
        |------------|-------------|------------------|
        | **SGD** (Descente de Gradient Stochastique) | Descente de gradient basique avec échantillons aléatoires | Simple, peut osciller |
        | **Momentum** | Ajoute de la vélocité aux mises à jour du gradient | Convergence plus rapide |
        | **Adam** | Taux d'apprentissage adaptatif par paramètre | Le plus populaire, fonctionne bien |
        | **RMSprop** | Adapte le taux d'apprentissage basé sur les gradients récents | Bon pour les RNN |
        
        ### Visualisation de la Descente de Gradient
        
        Imaginez faire rouler une balle sur une colline pour trouver le point le plus bas (perte minimale) :
        
        - **Taux d'Apprentissage** : La taille de chaque pas
        - **Gradient** : La direction de la descente la plus raide
        - **Momentum** : La vélocité de la balle due aux pas précédents
        
        """)
        
        st.warning("Choisir le bon optimiseur et le bon taux d'apprentissage est crucial. Un taux d'apprentissage trop élevé peut dépasser le minimum ; trop faible peut prendre une éternité à converger.")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Architecture du Réseau
        
        Les réseaux de neurones sont organisés en couches :
        
        1. **Couche d'Entrée** : Reçoit les données brutes
        2. **Couches Cachées** : Traitent l'information
        3. **Couche de Sortie** : Produit les prédictions
        
        Un réseau avec plusieurs couches cachées est appelé un **Réseau de Neurones Profond**.
        
        Chaque nœud du réseau représente un neurone et est connecté aux nœuds des couches adjacentes. 
        Ces connexions sont associées à des poids qui déterminent la force de la connexion.
        """)
    
    with col2:
        st.markdown("""
        ### Pourquoi les Réseaux de Neurones Fonctionnent
        
        La puissance des réseaux de neurones provient de :
        
        1. **Fonctions d'activation non-linéaires** : Permettent d'apprendre des motifs complexes
        2. **Couches multiples** : Permettent l'apprentissage hiérarchique de caractéristiques
        3. **Apprentissage basé sur le gradient** : Ajustent automatiquement les poids
        4. **Approximation universelle** : Peuvent approximer toute fonction continue
        
        Leur capacité à apprendre à partir de grands ensembles de données leur permet de bien généraliser sur de nouvelles données non vues.
        """)
    
    st.markdown("""
    ### Le Processus d'Apprentissage
    
    Les réseaux de neurones apprennent à travers un processus appelé **entraînement** :
    
    1. **Propagation Avant** : L'entrée circule à travers le réseau pour produire une sortie
    2. **Calcul de la Perte** : Comparer la prédiction avec la réponse réelle
    3. **Rétropropagation** : Calculer comment chaque poids a contribué à l'erreur
    4. **Descente de Gradient** : Ajuster les poids pour réduire l'erreur
    
    Ce processus se répète des milliers de fois jusqu'à ce que le réseau apprenne les motifs dans les données.
    La **technique de rétropropagation** améliore le réseau en propageant les signaux d'erreur en arrière 
    à travers les couches, permettant à chaque poids d'être ajusté proportionnellement à sa contribution à l'erreur.
    """)


def show_activation_functions():
    st.header("Fonctions d'Activation")
    
    st.markdown("""
    ## Pourquoi Avons-Nous Besoin de Fonctions d'Activation ?
    
    Sans fonctions d'activation, un réseau de neurones ne serait qu'une **transformation linéaire** :
    
    $$y = W_2 \\cdot (W_1 \\cdot x + b_1) + b_2 = W_{combined} \\cdot x + b_{combined}$$
    
    Peu importe le nombre de couches empilées, le résultat resterait linéaire ! Les fonctions d'activation 
    introduisent la **non-linéarité**, permettant aux réseaux d'apprendre des motifs complexes.
    """)
    
    viz_data = visualize_activations()
    x = viz_data['x']
    
    activation_choice = st.selectbox(
        "Sélectionnez une Fonction d'Activation à Explorer :",
        ["Sigmoïde", "Tanh", "ReLU", "Leaky ReLU", "Toutes les Fonctions"]
    )
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    if activation_choice == "Sigmoïde":
        axes[0].plot(x, viz_data['sigmoid'], 'b-', linewidth=2)
        axes[0].set_title('Sigmoïde : σ(z) = 1/(1 + e^(-z))')
        axes[1].plot(x, viz_data['sigmoid_derivative'], 'r-', linewidth=2)
        axes[1].set_title("Dérivée Sigmoïde : σ'(z) = σ(z)(1 - σ(z))")
        
        st.markdown("""
        ### Fonction Sigmoïde
        
        **Formule** : $\\sigma(z) = \\frac{1}{1 + e^{-z}}$
        
        **Plage de Sortie** : (0, 1)
        
        **Avantages** :
        - Sorties interprétables comme des probabilités
        - Lisse, dérivable partout
        
        **Inconvénients** :
        - Gradient évanescent pour de grandes valeurs de |z|
        - Non centrée sur zéro
        - Coûteuse en calcul
        
        **Cas d'Utilisation** : Sortie de classification binaire, portes LSTM/GRU
        """)
        
    elif activation_choice == "Tanh":
        axes[0].plot(x, viz_data['tanh'], 'b-', linewidth=2)
        axes[0].set_title('Tanh : tanh(z)')
        axes[1].plot(x, viz_data['tanh_derivative'], 'r-', linewidth=2)
        axes[1].set_title("Dérivée Tanh : 1 - tanh²(z)")
        
        st.markdown("""
        ### Tangente Hyperbolique (Tanh)
        
        **Formule** : $\\tanh(z) = \\frac{e^z - e^{-z}}{e^z + e^{-z}}$
        
        **Plage de Sortie** : (-1, 1)
        
        **Avantages** :
        - Sortie centrée sur zéro (meilleur flux de gradient)
        - Gradients plus forts que la sigmoïde
        
        **Inconvénients** :
        - Souffre toujours du problème de gradient évanescent
        - Coûteuse en calcul
        
        **Cas d'Utilisation** : Couches cachées (quand le centrage sur zéro est important), RNN
        """)
        
    elif activation_choice == "ReLU":
        axes[0].plot(x, viz_data['relu'], 'b-', linewidth=2)
        axes[0].set_title('ReLU : max(0, z)')
        axes[1].plot(x, viz_data['relu_derivative'], 'r-', linewidth=2)
        axes[1].set_title("Dérivée ReLU")
        
        st.markdown("""
        ### Unité Linéaire Rectifiée (ReLU)
        
        **Formule** : $\\text{ReLU}(z) = \\max(0, z)$
        
        **Plage de Sortie** : [0, ∞)
        
        **Avantages** :
        - Pas de gradient évanescent pour les valeurs positives
        - Efficace en calcul
        - Activation éparse (beaucoup de zéros)
        - Convergence plus rapide
        
        **Inconvénients** :
        - Problème du « ReLU mourant » (les neurones peuvent rester bloqués à 0)
        - Non centrée sur zéro
        
        **Cas d'Utilisation** : La plus populaire pour les couches cachées dans les réseaux modernes
        """)
        
    elif activation_choice == "Leaky ReLU":
        axes[0].plot(x, viz_data['leaky_relu'], 'b-', linewidth=2)
        axes[0].set_title('Leaky ReLU')
        axes[1].plot(x, viz_data['leaky_relu_derivative'], 'r-', linewidth=2)
        axes[1].set_title("Dérivée Leaky ReLU")
        
        st.markdown("""
        ### Leaky ReLU
        
        **Formule** : $\\text{LeakyReLU}(z) = \\begin{cases} z & \\text{si } z > 0 \\\\ \\alpha z & \\text{si } z \\leq 0 \\end{cases}$
        
        Où α est généralement égal à 0.01
        
        **Plage de Sortie** : (-∞, ∞)
        
        **Avantages** :
        - Résout le problème du « ReLU mourant »
        - Permet le flux de gradient pour les entrées négatives
        
        **Cas d'Utilisation** : Réseaux profonds où le ReLU mourant est préoccupant
        """)
        
    else:
        axes[0].plot(x, viz_data['sigmoid'], label='Sigmoïde')
        axes[0].plot(x, viz_data['tanh'], label='Tanh')
        axes[0].plot(x, viz_data['relu'], label='ReLU')
        axes[0].plot(x, viz_data['leaky_relu'], label='Leaky ReLU')
        axes[0].legend()
        axes[0].set_title('Toutes les Fonctions d\'Activation')
        
        axes[1].plot(x, viz_data['sigmoid_derivative'], label='Sigmoïde')
        axes[1].plot(x, viz_data['tanh_derivative'], label='Tanh')
        axes[1].plot(x, viz_data['relu_derivative'], label='ReLU')
        axes[1].plot(x, viz_data['leaky_relu_derivative'], label='Leaky ReLU')
        axes[1].legend()
        axes[1].set_title('Toutes les Dérivées')
    
    for ax in axes:
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.axvline(x=0, color='k', linewidth=0.5)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('z')
        ax.set_ylabel('f(z)')
    
    st.pyplot(fig)
    plt.close()


def show_perceptron():
    st.header("Le Perceptron")
    
    st.markdown("""
    ## Le Réseau de Neurones le Plus Simple
    
    Le **Perceptron**, inventé par Frank Rosenblatt en 1957, est le bloc de construction 
    fondamental des réseaux de neurones. C'est un seul neurone artificiel qui peut apprendre 
    à classifier des motifs **linéairement séparables**.
    
    ### Architecture
    
    ```
    [x₁] ──w₁──┐
    [x₂] ──w₂──┼──[Σ + b]──[échelon]──> ŷ
    [x₃] ──w₃──┘
    ```
    
    ### Modèle Mathématique
    
    1. **Combinaison Linéaire** : $z = \\sum_{i} w_i x_i + b$
    2. **Activation Échelon** : $\\hat{y} = \\begin{cases} 1 & \\text{si } z \\geq 0 \\\\ 0 & \\text{si } z < 0 \\end{cases}$
    
    ### Règle d'Apprentissage
    
    L'algorithme d'apprentissage du perceptron met à jour les poids quand une erreur est commise :
    
    $$w_{nouveau} = w_{ancien} + \\alpha \\cdot (y_{vrai} - y_{prédit}) \\cdot x$$
    $$b_{nouveau} = b_{ancien} + \\alpha \\cdot (y_{vrai} - y_{prédit})$$
    
    Où α est le taux d'apprentissage.
    """)
    
    st.subheader("Démo Interactive : Portes Logiques")
    
    col1, col2 = st.columns([1, 2])
    
    gate_labels = {
        "ET": "AND",
        "OU": "OR",
        "NON-ET": "NAND",
        "OU Exclusif": "XOR"
    }
    
    with col1:
        gate_display = st.selectbox(
            "Sélectionnez une Porte Logique :",
            ["ET", "OU", "NON-ET", "OU Exclusif"]
        )
        gate_choice = gate_labels[gate_display]
        
        learning_rate = st.slider(
            "Taux d'Apprentissage :",
            min_value=0.01,
            max_value=1.0,
            value=0.1,
            step=0.01
        )
        
        epochs = st.slider(
            "Époques d'Entraînement :",
            min_value=10,
            max_value=200,
            value=100,
            step=10
        )
        
        train_btn = st.button("Entraîner le Perceptron")
    
    with col2:
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        gates = {
            'AND': np.array([0, 0, 0, 1]),
            'OR': np.array([0, 1, 1, 1]),
            'NAND': np.array([1, 1, 1, 0]),
            'XOR': np.array([0, 1, 1, 0])
        }
        y = gates[gate_choice]
        
        if train_btn or 'perceptron_result' not in st.session_state:
            perceptron = Perceptron(n_inputs=2, learning_rate=learning_rate)
            history = perceptron.train(X, y, epochs=epochs, verbose=False)
            predictions = perceptron.predict(X)
            
            st.session_state.perceptron_result = {
                'predictions': predictions,
                'history': history,
                'weights': perceptron.weights.copy(),
                'bias': perceptron.bias,
                'perceptron': perceptron
            }
        
        result = st.session_state.perceptron_result
        
        st.markdown(f"### Résultats de la Porte {gate_display}")
        
        results_df = {
            'x₁': X[:, 0],
            'x₂': X[:, 1],
            'Attendu': y,
            'Prédit': result['predictions'],
            'Correct': ['Oui' if y[i] == result['predictions'][i] else 'Non' for i in range(4)]
        }
        st.dataframe(results_df, use_container_width=True)
        
        accuracy = np.mean(y == result['predictions'])
        
        if accuracy == 1.0:
            st.success(f"Classification parfaite ! Précision : {accuracy:.0%}")
        else:
            st.warning(f"Précision : {accuracy:.0%}")
            if gate_choice == "XOR":
                st.error("Le perceptron ne peut pas apprendre OU Exclusif - ce n'est pas linéairement séparable !")
    
    if gate_choice == "XOR":
        st.markdown("""
        ## Le Problème OU Exclusif
        
        La porte OU Exclusif produit 1 quand les entrées sont **différentes**, 0 quand elles sont **identiques**.
        
        **Pourquoi le Perceptron Échoue sur OU Exclusif :**
        
        Un perceptron crée une **frontière de décision linéaire** (une ligne droite en 2D). 
        OU Exclusif nécessite une frontière **non-linéaire** - vous ne pouvez pas tracer une seule ligne droite 
        pour séparer les classes !
        
        ```
          x₂
          │
        1 ├───●(0,1)───────●(1,1)
          │   classe 1     classe 0
          │
        0 ├───●(0,0)───────●(1,0)
          │   classe 0     classe 1
          └───────────────────── x₁
              0             1
        ```
        
        **Solution** : Perceptron Multicouche (MLP) avec des couches cachées !
        """)


def show_mlp():
    st.header("Perceptron Multicouche (MLP)")
    
    st.markdown("""
    ## Du Perceptron aux Réseaux Profonds
    
    Le **Perceptron Multicouche** étend le perceptron simple en ajoutant des **couches cachées** 
    entre l'entrée et la sortie. Cela permet d'apprendre des motifs complexes et non-linéaires.
    
    ### Architecture du Réseau
    
    ```
    Couche d'Entrée  Couche Cachée 1   Couche Cachée 2   Couche de Sortie
    (4 neurones)     (10 neurones)     (8 neurones)      (3 neurones)
    
        [x₁]              [h₁]              [h₁]             [ŷ₁]
        [x₂]  ────W¹────  [h₂]  ────W²────  [h₂]  ────W³────  [ŷ₂]
        [x₃]              ...               ...               [ŷ₃]
        [x₄]              [h₁₀]             [h₈]
    ```
    
    ### Propagation Avant
    
    Pour chaque couche $l$ de 1 à L :
    
    $$Z^{[l]} = W^{[l]} \\cdot A^{[l-1]} + b^{[l]}$$
    $$A^{[l]} = f^{[l]}(Z^{[l]})$$
    
    Où :
    - $W^{[l]}$ est la matrice de poids pour la couche $l$
    - $A^{[0]} = X$ (données d'entrée)
    - $f^{[l]}$ est la fonction d'activation (ReLU pour les couches cachées, Softmax pour la sortie)
    """)
    
    st.markdown("""
    ### Rétropropagation
    
    La **règle de la chaîne** nous permet de calculer les gradients pour tous les paramètres :
    
    **Couche de Sortie :**
    $$dZ^{[L]} = A^{[L]} - Y$$
    $$dW^{[L]} = \\frac{1}{m} dZ^{[L]} \\cdot A^{[L-1]T}$$
    
    **Couches Cachées** (pour $l = L-1, ..., 1$) :
    $$dA^{[l]} = W^{[l+1]T} \\cdot dZ^{[l+1]}$$
    $$dZ^{[l]} = dA^{[l]} \\odot f'^{[l]}(Z^{[l]})$$
    $$dW^{[l]} = \\frac{1}{m} dZ^{[l]} \\cdot A^{[l-1]T}$$
    
    ### Mise à Jour des Poids (Descente de Gradient)
    
    $$W^{[l]} := W^{[l]} - \\alpha \\cdot dW^{[l]}$$
    $$b^{[l]} := b^{[l]} - \\alpha \\cdot db^{[l]}$$
    """)
    
    st.subheader("Démo : MLP Résolvant OU Exclusif")
    
    if st.button("Résoudre OU Exclusif avec MLP"):
        with st.spinner("Entraînement du MLP sur OU Exclusif..."):
            result = solve_xor_with_mlp()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Table de Vérité OU Exclusif")
            results_df = {
                'x₁': result['X'][:, 0],
                'x₂': result['X'][:, 1],
                'Attendu': result['y'],
                'Prédit': result['predictions'],
                'Correct': ['Oui' if result['y'][i] == result['predictions'][i] else 'Non' for i in range(4)]
            }
            st.dataframe(results_df, use_container_width=True)
            st.success(f"Précision : {result['accuracy']:.0%}")
        
        with col2:
            st.markdown("### Courbes d'Entraînement")
            fig, axes = plt.subplots(1, 2, figsize=(10, 3))
            
            axes[0].plot(result['history']['loss'])
            axes[0].set_title('Perte au Cours des Époques')
            axes[0].set_xlabel('Époque')
            axes[0].set_ylabel('Perte Entropie Croisée')
            axes[0].grid(True, alpha=0.3)
            
            axes[1].plot(result['history']['accuracy'])
            axes[1].set_title('Précision au Cours des Époques')
            axes[1].set_xlabel('Époque')
            axes[1].set_ylabel('Précision')
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        st.markdown("""
        Le MLP avec juste **une couche cachée** apprend OU Exclusif avec succès ! 
        Cela démontre la puissance des couches cachées pour apprendre des motifs non-linéaires.
        """)


def show_iris_combined():
    st.header("Prédiction et Reconnaissance Iris")
    
    st.markdown("""
    ## Classification des Espèces d'Iris
    
    Cette section combine la prédiction basée sur les mesures et la reconnaissance d'images pour classifier les fleurs d'iris.
    Choisissez votre méthode préférée ci-dessous.
    """)
    
    mode_tabs = st.tabs([
        "Prédiction par Mesures",
        "Reconnaissance d'Images"
    ])
    
    with mode_tabs[0]:
        show_iris_prediction_content()
    
    with mode_tabs[1]:
        show_image_recognition_content()


def show_iris_prediction_content():
    st.subheader("Prédiction des Espèces d'Iris")
    
    st.markdown("""
    ### Prédire les Espèces d'Iris
    
    Utilisez un réseau de neurones entraîné pour classifier les fleurs d'iris en fonction de leurs mesures.
    Choisissez entre la saisie de mesure unique, le téléchargement CSV par lot ou le téléchargement multi-images pour la classification.
    """)
    
    saved_models = get_saved_model_names()
    iris_models = [m for m in saved_models if 'iris' in m[0].lower()]
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("État du Modèle")
        
        if st.button("Entraîner le Classifieur Iris", type="primary"):
            with st.spinner("Entraînement du modèle..."):
                X_train, X_test, y_train, y_test, feature_names, class_names = load_iris_dataset(source='sklearn')
                
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                mlp = MLP(
                    layer_sizes=[4, 32, 16, 3],
                    learning_rate=0.01,
                    activation='relu',
                    weight_init='he'
                )
                
                mlp.fit(X_train_scaled, y_train, epochs=800, batch_size=16, verbose=False)
                
                y_pred = mlp.predict(X_test_scaled)
                accuracy = np.mean(y_pred == y_test)
                
                save_trained_model(
                    name='iris_classifier',
                    model=mlp,
                    scaler_mean=scaler.mean_,
                    scaler_std=scaler.scale_,
                    accuracy=accuracy,
                    class_names=list(class_names)
                )
                
                st.success(f"Modèle entraîné et sauvegardé ! Précision de test : {accuracy:.1%}")
                st.rerun()
        
        if iris_models:
            st.success(f"Modèle Actif : {iris_models[0][0]} (Précision : {iris_models[0][1]:.1%})")
        else:
            st.warning("Aucun modèle entraîné trouvé. Cliquez d'abord sur 'Entraîner le Classifieur Iris'.")
    
    with col2:
        st.subheader("Référence des Fleurs d'Iris")
        iris_images = sorted([f for f in os.listdir('attached_assets') if f.startswith('iris-') and f.endswith('.jpg')])
        if iris_images[:6]:
            cols = st.columns(3)
            for idx, img_file in enumerate(iris_images[:6]):
                with cols[idx % 3]:
                    st.image(f"attached_assets/{img_file}", use_container_width=True)
    
    st.markdown("---")
    
    prediction_mode = st.radio(
        "Mode de Prédiction :",
        ["Mesure Unique", "Téléchargement CSV par Lot", "Téléchargement Multi-Images"],
        horizontal=True
    )
    
    if prediction_mode == "Mesure Unique":
        st.subheader("Entrez les Mesures de la Fleur")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            sepal_length = st.number_input("Longueur du Sépale (cm)", min_value=0.0, max_value=10.0, value=5.1, step=0.1)
        with col2:
            sepal_width = st.number_input("Largeur du Sépale (cm)", min_value=0.0, max_value=10.0, value=3.5, step=0.1)
        with col3:
            petal_length = st.number_input("Longueur du Pétale (cm)", min_value=0.0, max_value=10.0, value=1.4, step=0.1)
        with col4:
            petal_width = st.number_input("Largeur du Pétale (cm)", min_value=0.0, max_value=10.0, value=0.2, step=0.1)
        
        if st.button("Prédire l'Espèce", key="single_predict"):
            model_data = load_trained_model('iris_classifier')
            
            if model_data is None:
                st.error("Aucun modèle entraîné trouvé. Veuillez d'abord entraîner le modèle.")
            else:
                mlp = MLP.from_saved_weights(
                    layer_sizes=model_data['layer_sizes'],
                    activation=model_data['activation'],
                    weights=model_data['weights'],
                    biases=model_data['biases']
                )
                
                features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
                
                if model_data['scaler_mean'] is not None:
                    features = (features - model_data['scaler_mean']) / model_data['scaler_std']
                
                prediction = mlp.predict(features)[0]
                probabilities = mlp.predict_proba(features)[0]
                
                class_names = model_data['class_names']
                predicted_class = class_names[prediction]
                
                st.success(f"**Espèce Prédite : {predicted_class}**")
                
                st.markdown("#### Scores de Confiance :")
                prob_cols = st.columns(3)
                colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
                for i, (name, prob) in enumerate(zip(class_names, probabilities)):
                    with prob_cols[i]:
                        st.metric(name, f"{prob:.1%}")
                        st.progress(float(prob))
    
    elif prediction_mode == "Téléchargement CSV par Lot":
        st.subheader("Télécharger un CSV pour la Prédiction par Lot")
        
        st.markdown("""
        **Format CSV Requis :**
        ```
        Id,SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm,Species
        1,5.1,3.5,1.4,0.2,Iris-setosa
        2,4.9,3.0,1.4,0.2,Iris-setosa
        ...
        ```
        La colonne `Species` est optionnelle - si fournie, elle sera utilisée pour calculer la précision.
        """)
        
        uploaded_csv = st.file_uploader("Télécharger un fichier CSV", type=['csv'], key="csv_uploader")
        
        if uploaded_csv is not None:
            try:
                df = pd.read_csv(uploaded_csv)
                st.write(f"**{len(df)} échantillons chargés**")
                st.dataframe(df.head(10), use_container_width=True)
                
                required_cols = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    st.error(f"Colonnes requises manquantes : {missing_cols}")
                else:
                    if st.button("Exécuter la Prédiction par Lot", type="primary"):
                        model_data = load_trained_model('iris_classifier')
                        
                        if model_data is None:
                            st.error("Aucun modèle entraîné trouvé. Veuillez d'abord entraîner le modèle.")
                        else:
                            mlp = MLP.from_saved_weights(
                                layer_sizes=model_data['layer_sizes'],
                                activation=model_data['activation'],
                                weights=model_data['weights'],
                                biases=model_data['biases']
                            )
                            
                            features = df[required_cols].values
                            
                            if model_data['scaler_mean'] is not None:
                                features = (features - model_data['scaler_mean']) / model_data['scaler_std']
                            
                            predictions = mlp.predict(features)
                            probabilities = mlp.predict_proba(features)
                            
                            class_names = model_data['class_names']
                            predicted_classes = [class_names[p] for p in predictions]
                            
                            results_df = df.copy()
                            results_df['Predicted_Species'] = predicted_classes
                            results_df['Confidence'] = [f"{max(prob)*100:.1f}%" for prob in probabilities]
                            
                            for i, name in enumerate(class_names):
                                results_df[f'Prob_{name}'] = [f"{prob[i]*100:.1f}%" for prob in probabilities]
                            
                            if 'Species' in df.columns:
                                species_map = {
                                    'Iris-setosa': 'setosa',
                                    'Iris-versicolor': 'versicolor', 
                                    'Iris-virginica': 'virginica',
                                    'setosa': 'setosa',
                                    'versicolor': 'versicolor',
                                    'virginica': 'virginica'
                                }
                                actual_mapped = df['Species'].map(species_map).fillna(df['Species'])
                                results_df['Correct'] = actual_mapped == results_df['Predicted_Species']
                                accuracy = results_df['Correct'].mean()
                                
                                st.success(f"Prédiction par Lot Terminée ! Précision : {accuracy:.1%}")
                            else:
                                st.success("Prédiction par Lot Terminée !")
                            
                            st.subheader("Résultats de Prédiction")
                            st.dataframe(results_df, use_container_width=True)
                            
                            st.subheader("Résumé des Prédictions")
                            summary_counts = pd.Series(predicted_classes).value_counts()
                            
                            col1, col2 = st.columns([1, 1])
                            with col1:
                                st.markdown("**Distribution des Espèces :**")
                                for species, count in summary_counts.items():
                                    st.write(f"- {species}: {count} ({count/len(predictions)*100:.1f}%)")
                            
                            with col2:
                                fig, ax = plt.subplots(figsize=(6, 4))
                                colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
                                ax.pie(summary_counts.values, labels=summary_counts.index, autopct='%1.1f%%', colors=colors[:len(summary_counts)])
                                ax.set_title('Distribution des Espèces Prédites')
                                st.pyplot(fig)
                                plt.close()
                            
                            csv_buffer = io.StringIO()
                            results_df.to_csv(csv_buffer, index=False)
                            st.download_button(
                                label="Télécharger les Résultats en CSV",
                                data=csv_buffer.getvalue(),
                                file_name="iris_predictions.csv",
                                mime="text/csv"
                            )
                            
            except Exception as e:
                st.error(f"Erreur lors de la lecture du CSV : {str(e)}")
    
    else:
        st.subheader("Télécharger Plusieurs Images de Fleurs d'Iris")
        st.info("Téléchargez plusieurs images de fleurs d'iris pour la classification. Le modèle analysera chaque image pour aider à identifier l'espèce.")
        
        uploaded_files = st.file_uploader(
            "Choisissez des images de fleurs d'iris", 
            type=['jpg', 'jpeg', 'png'],
            accept_multiple_files=True,
            key="multi_image_uploader"
        )
        
        if uploaded_files:
            st.write(f"**{len(uploaded_files)} images téléchargées**")
            
            cols_per_row = 4
            for i in range(0, len(uploaded_files), cols_per_row):
                cols = st.columns(cols_per_row)
                for j, col in enumerate(cols):
                    if i + j < len(uploaded_files):
                        with col:
                            st.image(uploaded_files[i + j], caption=f"Image {i+j+1}", use_container_width=True)
            
            st.markdown("---")
            st.subheader("Saisie Manuelle de Classification")
            st.markdown("""
            Comme la classification basée sur les images nécessite un modèle CNN, veuillez entrer les mesures pour chaque image téléchargée.
            Cela aide à corréler les caractéristiques visuelles avec les données de mesure.
            """)
            
            if 'image_measurements' not in st.session_state:
                st.session_state.image_measurements = []
            
            for idx, uploaded_file in enumerate(uploaded_files):
                with st.expander(f"Entrer les mesures pour l'Image {idx + 1} : {uploaded_file.name}"):
                    cols = st.columns([1, 1, 1, 1, 1])
                    with cols[0]:
                        st.image(uploaded_file, use_container_width=True)
                    with cols[1]:
                        sl = st.number_input(f"Longueur du Sépale", min_value=0.0, max_value=10.0, value=5.0, step=0.1, key=f"sl_{idx}")
                    with cols[2]:
                        sw = st.number_input(f"Largeur du Sépale", min_value=0.0, max_value=10.0, value=3.0, step=0.1, key=f"sw_{idx}")
                    with cols[3]:
                        pl = st.number_input(f"Longueur du Pétale", min_value=0.0, max_value=10.0, value=1.5, step=0.1, key=f"pl_{idx}")
                    with cols[4]:
                        pw = st.number_input(f"Largeur du Pétale", min_value=0.0, max_value=10.0, value=0.2, step=0.1, key=f"pw_{idx}")
            
            if st.button("Classifier Toutes les Images", type="primary"):
                model_data = load_trained_model('iris_classifier')
                
                if model_data is None:
                    st.error("Aucun modèle entraîné trouvé. Veuillez d'abord entraîner le modèle.")
                else:
                    mlp = MLP.from_saved_weights(
                        layer_sizes=model_data['layer_sizes'],
                        activation=model_data['activation'],
                        weights=model_data['weights'],
                        biases=model_data['biases']
                    )
                    
                    results = []
                    for idx, uploaded_file in enumerate(uploaded_files):
                        sl = st.session_state.get(f"sl_{idx}", 5.0)
                        sw = st.session_state.get(f"sw_{idx}", 3.0)
                        pl = st.session_state.get(f"pl_{idx}", 1.5)
                        pw = st.session_state.get(f"pw_{idx}", 0.2)
                        
                        features = np.array([[sl, sw, pl, pw]])
                        
                        if model_data['scaler_mean'] is not None:
                            features = (features - model_data['scaler_mean']) / model_data['scaler_std']
                        
                        prediction = mlp.predict(features)[0]
                        probabilities = mlp.predict_proba(features)[0]
                        
                        class_names = model_data['class_names']
                        predicted_class = class_names[prediction]
                        confidence = max(probabilities)
                        
                        results.append({
                            'image_name': uploaded_file.name,
                            'predicted_species': predicted_class,
                            'confidence': confidence,
                            'probabilities': probabilities
                        })
                    
                    st.success(f"{len(results)} images classifiées !")
                    
                    st.subheader("Résultats de Classification")
                    cols_per_row = 3
                    for i in range(0, len(results), cols_per_row):
                        cols = st.columns(cols_per_row)
                        for j, col in enumerate(cols):
                            if i + j < len(results):
                                result = results[i + j]
                                with col:
                                    st.image(uploaded_files[i + j], use_container_width=True)
                                    st.markdown(f"**{result['predicted_species']}**")
                                    st.write(f"Confiance : {result['confidence']:.1%}")
                    
                    results_df = pd.DataFrame([{
                        'Image': r['image_name'],
                        'Espèce Prédite': r['predicted_species'],
                        'Confiance': f"{r['confidence']*100:.1f}%"
                    } for r in results])
                    
                    csv_buffer = io.StringIO()
                    results_df.to_csv(csv_buffer, index=False)
                    st.download_button(
                        label="Télécharger les Résultats en CSV",
                        data=csv_buffer.getvalue(),
                        file_name="resultats_classification_images.csv",
                        mime="text/csv"
                    )
            
            st.markdown("---")
            st.markdown("""
            **Note :** Pour une classification d'images automatisée, un Réseau de Neurones Convolutif (CNN) serait nécessaire :
            - Un CNN peut apprendre les caractéristiques visuelles directement à partir des images
            - L'apprentissage par transfert à partir de modèles pré-entraînés (ResNet, VGG) pourrait être appliqué
            - Un grand ensemble de données d'images de fleurs d'iris étiquetées améliorerait la précision
            """)


def show_image_recognition_content():
    st.subheader("Reconnaissance d'Images Iris")
    
    st.markdown("""
    ### Classification d'Iris Basée sur l'Image
    
    Cette fonctionnalité vous permet de classifier des fleurs d'iris directement à partir d'images en utilisant des réseaux de neurones.
    Le système extrait les caractéristiques visuelles (couleurs, formes, textures) des images et utilise un
    MLP entraîné pour identifier l'espèce d'iris.
    
    **Trois Espèces d'Iris :**
    - **Iris-setosa** - Connu pour ses pétales plus petits
    - **Iris-versicolor** - Pétales de taille moyenne  
    - **Iris-virginica** - Pétales plus grands
    """)
    
    mode_tabs = st.tabs(["Classifier une Image", "Gérer les Images de Référence", "Entraîner le Modèle Image"])
    
    with mode_tabs[0]:
        show_image_classification()
    
    with mode_tabs[1]:
        show_reference_management()
    
    with mode_tabs[2]:
        show_image_model_training()


def show_image_classification():
    st.subheader("Classifier un Iris à partir d'une Image")
    
    model_data = load_iris_image_model('iris_image_classifier')
    
    if model_data is None:
        st.warning("Aucun classificateur d'images entraîné trouvé. Veuillez d'abord entraîner un modèle dans l'onglet 'Entraîner le Modèle Image'.")
        st.info("Vous devez télécharger des images de référence et entraîner le modèle avant la classification.")
        return
    
    st.success(f"Classificateur d'images chargé - Précision : {model_data['accuracy']:.1%} (entraîné sur {model_data['num_reference_images']} images)")
    
    input_method = st.radio("Choisissez la méthode d'entrée :", ["Télécharger une Image", "Capture Caméra"], horizontal=True)
    
    image_data = None
    
    if input_method == "Télécharger une Image":
        uploaded_file = st.file_uploader(
            "Télécharger une image de fleur d'iris",
            type=['jpg', 'jpeg', 'png', 'webp'],
            help="Téléchargez une photo claire d'une fleur d'iris pour la classification"
        )
        if uploaded_file is not None:
            image_data = uploaded_file.read()
            st.image(image_data, caption="Image Téléchargée", width=300)
    else:
        camera_image = st.camera_input("Prendre une photo d'une fleur d'iris")
        if camera_image is not None:
            image_data = camera_image.read()
    
    if image_data is not None:
        if st.button("Classifier l'Espèce d'Iris", type="primary"):
            with st.spinner("Analyse de l'image..."):
                try:
                    features = extract_features(image_data)
                    
                    features_normalized, _, _ = normalize_features(
                        features.reshape(1, -1),
                        mean=model_data['feature_scaler_mean'],
                        std=model_data['feature_scaler_std']
                    )
                    
                    mlp = MLP(
                        layer_sizes=model_data['layer_sizes'],
                        activation=model_data['activation']
                    )
                    mlp.weights = model_data['weights']
                    mlp.biases = model_data['biases']
                    
                    X_T = features_normalized.T
                    A, _ = mlp.forward_propagation(X_T)
                    probabilities = A[mlp.L][:, 0]
                    
                    predicted_class = np.argmax(probabilities)
                    confidence = probabilities[predicted_class]
                    species_name = SPECIES_MAPPING[predicted_class]
                    
                    st.markdown("---")
                    st.subheader("Résultat de Classification")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Espèce Prédite", species_name)
                    with col2:
                        st.metric("Confiance", f"{confidence:.1%}")
                    with col3:
                        st.metric("ID Espèce", predicted_class)
                    
                    st.markdown("### Distribution des Probabilités")
                    prob_df = pd.DataFrame({
                        'Espèce': [SPECIES_MAPPING[i] for i in range(3)],
                        'Probabilité': probabilities
                    })
                    
                    fig, ax = plt.subplots(figsize=(8, 4))
                    colors = ['#2ecc71' if i == predicted_class else '#3498db' for i in range(3)]
                    ax.barh(prob_df['Espèce'], prob_df['Probabilité'], color=colors)
                    ax.set_xlim(0, 1)
                    ax.set_xlabel('Probabilité')
                    ax.set_title('Distribution des Probabilités par Espèce')
                    for i, (species, prob) in enumerate(zip(prob_df['Espèce'], prob_df['Probabilité'])):
                        ax.text(prob + 0.02, i, f'{prob:.1%}', va='center')
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                    
                except Exception as e:
                    st.error(f"Erreur lors de la classification de l'image : {str(e)}")


def show_reference_management():
    st.subheader("Gérer les Images de Référence")
    
    st.markdown("""
    Téléchargez des images de référence pour chaque espèce d'iris. Ces images seront utilisées pour entraîner
    le classificateur d'images. Pour de meilleurs résultats, téléchargez au moins 5-10 images par espèce.
    """)
    
    ref_counts = get_reference_image_count()
    if ref_counts:
        st.markdown("### Nombre d'Images de Référence Actuelles")
        cols = st.columns(3)
        for idx, (species, count) in enumerate(ref_counts.items()):
            with cols[idx % 3]:
                st.metric(species, f"{count} images")
    
    st.markdown("---")
    st.markdown("### Télécharger de Nouvelles Images de Référence")
    
    species_choice = st.selectbox(
        "Sélectionner l'Espèce d'Iris :",
        options=list(SPECIES_MAPPING.values()),
        help="Choisissez l'espèce pour les images que vous téléchargez"
    )
    species_id = SPECIES_ID_MAPPING[species_choice]
    
    uploaded_files = st.file_uploader(
        f"Télécharger des images {species_choice}",
        type=['jpg', 'jpeg', 'png', 'webp'],
        accept_multiple_files=True,
        help="Téléchargez des photos claires de cette espèce d'iris"
    )
    
    if uploaded_files:
        st.write(f"{len(uploaded_files)} images sélectionnées")
        
        preview_cols = st.columns(min(len(uploaded_files), 5))
        for idx, file in enumerate(uploaded_files[:5]):
            with preview_cols[idx]:
                st.image(file, use_container_width=True)
        
        if st.button("Sauvegarder les Images de Référence", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            saved_count = 0
            for idx, file in enumerate(uploaded_files):
                try:
                    image_bytes = file.read()
                    file.seek(0)
                    
                    os.makedirs("reference_images", exist_ok=True)
                    image_path = f"reference_images/{species_choice}_{int(time.time())}_{idx}.png"
                    
                    img = Image.open(io.BytesIO(image_bytes))
                    img.save(image_path)
                    
                    image_id = save_iris_reference_image(
                        species_id=species_id,
                        species_name=species_choice,
                        image_path=image_path,
                        filename=file.name
                    )
                    
                    features = extract_features(image_bytes)
                    save_iris_image_features(
                        reference_image_id=image_id,
                        species_id=species_id,
                        species_name=species_choice,
                        feature_vector=features
                    )
                    
                    saved_count += 1
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                    status_text.text(f"Traitement de l'image {idx + 1}/{len(uploaded_files)}")
                    
                except Exception as e:
                    st.error(f"Erreur lors du traitement de {file.name} : {str(e)}")
            
            st.success(f"{saved_count} images de référence sauvegardées avec succès avec les caractéristiques extraites !")
            st.rerun()
    
    st.markdown("---")
    st.markdown("### Voir les Images de Référence Existantes")
    
    view_species = st.selectbox(
        "Voir les images par espèce :",
        options=["Toutes"] + list(SPECIES_MAPPING.values()),
        key="view_species"
    )
    
    if view_species == "Toutes":
        ref_images = get_iris_reference_images()
    else:
        species_id = SPECIES_ID_MAPPING[view_species]
        ref_images = get_iris_reference_images(species_id)
    
    if ref_images:
        st.write(f"{len(ref_images)} images de référence trouvées")
        
        cols = st.columns(4)
        for idx, img_record in enumerate(ref_images[:12]):
            with cols[idx % 4]:
                if img_record['image_path'] and os.path.exists(img_record['image_path']):
                    st.image(img_record['image_path'], use_container_width=True)
                    st.caption(f"{img_record['species_name']}")
                    if st.button("Supprimer", key=f"del_{img_record['id']}"):
                        delete_iris_reference_image(img_record['id'])
                        if img_record['image_path'] and os.path.exists(img_record['image_path']):
                            os.remove(img_record['image_path'])
                        st.rerun()
    else:
        st.info("Aucune image de référence trouvée. Téléchargez des images ci-dessus.")


def show_image_model_training():
    st.subheader("Entraîner le Classificateur d'Images")
    
    st.markdown("""
    Entraînez un réseau de neurones pour classifier les espèces d'iris basé sur les caractéristiques d'image.
    Le modèle extrait les caractéristiques visuelles (couleurs, formes, textures) de vos
    images de référence et apprend à distinguer entre les espèces.
    """)
    
    feature_counts = get_feature_count()
    total_features = sum(feature_counts.values()) if feature_counts else 0
    
    st.markdown("### État des Données d'Entraînement")
    if feature_counts:
        cols = st.columns(3)
        for idx, (species, count) in enumerate(feature_counts.items()):
            with cols[idx % 3]:
                st.metric(species, f"{count} échantillons")
        
        st.write(f"**Total des échantillons d'entraînement :** {total_features}")
        
        min_per_class = min(feature_counts.values()) if feature_counts else 0
        if min_per_class < 3:
            st.warning("Besoin d'au moins 3 images par espèce pour un entraînement efficace. Veuillez télécharger plus d'images de référence.")
    else:
        st.warning("Aucune donnée d'entraînement disponible. Veuillez d'abord télécharger des images de référence.")
        return
    
    st.markdown("---")
    st.markdown("### Configuration du Modèle")
    
    col1, col2 = st.columns(2)
    
    with col1:
        hidden_size_1 = st.slider("Taille Couche Cachée 1", 16, 128, 64, 8)
        hidden_size_2 = st.slider("Taille Couche Cachée 2", 8, 64, 32, 8)
        learning_rate = st.select_slider(
            "Taux d'Apprentissage",
            options=[0.001, 0.005, 0.01, 0.05, 0.1],
            value=0.01
        )
    
    with col2:
        epochs = st.slider("Époques d'Entraînement", 100, 2000, 500, 100)
        activation = st.selectbox("Fonction d'Activation", ["relu", "sigmoid", "tanh"])
        batch_size = st.slider("Taille du Lot", 4, 32, 16, 4)
    
    if st.button("Entraîner le Classificateur d'Images", type="primary"):
        X, y, species_names = get_all_iris_image_features()
        
        if X is None or len(X) < 6:
            st.error("Pas assez de données d'entraînement. Veuillez télécharger plus d'images de référence.")
            return
        
        with st.spinner("Entraînement du classificateur d'images..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            indices = np.random.permutation(len(X))
            X = X[indices]
            y = y[indices]
            
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            X_train_norm, scaler_mean, scaler_std = normalize_features(X_train)
            X_test_norm, _, _ = normalize_features(X_test, mean=scaler_mean, std=scaler_std)
            
            feature_dim = X_train.shape[1]
            layer_sizes = [feature_dim, hidden_size_1, hidden_size_2, 3]
            
            mlp = MLP(
                layer_sizes=layer_sizes,
                learning_rate=learning_rate,
                activation=activation
            )
            
            status_text.text("Entraînement en cours...")
            mlp.fit(X_train_norm, y_train, epochs=epochs, batch_size=batch_size, verbose=False)
            
            progress_bar.progress(1.0)
            
            train_predictions = mlp.predict(X_train_norm)
            test_predictions = mlp.predict(X_test_norm)
            
            train_accuracy = np.mean(train_predictions == y_train)
            test_accuracy = np.mean(test_predictions == y_test)
            
            save_iris_image_model(
                name='iris_image_classifier',
                model=mlp,
                feature_scaler_mean=scaler_mean,
                feature_scaler_std=scaler_std,
                accuracy=test_accuracy,
                num_reference_images=total_features,
                feature_version='v1'
            )
            
            st.success("Modèle entraîné et sauvegardé avec succès !")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Précision Entraînement", f"{train_accuracy:.1%}")
            with col2:
                st.metric("Précision Test", f"{test_accuracy:.1%}")
            
            st.markdown("### Architecture du Modèle")
            st.write(f"Tailles des couches : {layer_sizes}")
            st.write(f"Total des paramètres : {sum(mlp.weights[l].size + mlp.biases[l].size for l in range(1, mlp.L + 1))}")
    
    existing_model = load_iris_image_model('iris_image_classifier')
    if existing_model:
        st.markdown("---")
        st.markdown("### Modèle Entraîné Actuel")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Précision", f"{existing_model['accuracy']:.1%}" if existing_model['accuracy'] else "N/A")
        with col2:
            st.metric("Images d'Entraînement", existing_model['num_reference_images'] or 0)
        with col3:
            st.metric("Créé le", existing_model['created_at'][:10] if existing_model['created_at'] else "N/A")


def show_iris_training():
    st.header("Entraîner le MLP sur le Jeu de Données Iris")
    
    st.markdown("""
    ## Le Jeu de Données Iris
    
    Un jeu de données classique pour la classification contenant 150 échantillons de fleurs d'iris :
    - **4 Caractéristiques** : longueur du sépale, largeur du sépale, longueur du pétale, largeur du pétale
    - **3 Classes** : setosa, versicolor, virginica (50 échantillons chacune)
    """)
    
    iris_images = sorted([f for f in os.listdir('attached_assets') if f.startswith('iris-') and f.endswith('.jpg')])
    if iris_images:
        with st.expander("Voir les Images de Fleurs Iris Setosa", expanded=False):
            st.markdown("### Échantillons Iris Setosa")
            st.markdown("Ce sont de vraies photographies de fleurs Iris Setosa - l'une des trois espèces du jeu de données.")
            cols = st.columns(6)
            for idx, img_file in enumerate(iris_images[:12]):
                with cols[idx % 6]:
                    st.image(f"attached_assets/{img_file}", use_container_width=True)
            if len(iris_images) > 12:
                cols2 = st.columns(6)
                for idx, img_file in enumerate(iris_images[12:]):
                    with cols2[idx % 6]:
                        st.image(f"attached_assets/{img_file}", use_container_width=True)
    
    db_sample_count = get_iris_sample_count()
    if db_sample_count > 0:
        st.info(f"La base de données contient {db_sample_count} échantillons Iris chargés depuis CSV.")
    
    st.sidebar.header("Source des Données")
    
    data_source = st.sidebar.radio(
        "Charger les données Iris depuis :",
        ["sklearn (intégré)", "Base de données (import CSV)"],
        help="Choisissez si vous voulez charger depuis sklearn ou la base de données"
    )
    
    if data_source == "Base de données (import CSV)" and db_sample_count == 0:
        if os.path.exists('attached_assets/Iris_1765047715889.csv'):
            if st.sidebar.button("Importer CSV dans la Base"):
                try:
                    count = import_iris_from_csv('attached_assets/Iris_1765047715889.csv')
                    st.sidebar.success(f"{count} échantillons importés dans la base de données !")
                    st.rerun()
                except Exception as e:
                    st.sidebar.error(f"Erreur d'importation : {e}")
        else:
            st.sidebar.warning("Aucun fichier CSV trouvé. Utilisation de sklearn à la place.")
            data_source = "sklearn (intégré)"
    
    st.sidebar.header("Configuration du Modèle")
    
    hidden_layer_1 = st.sidebar.slider(
        "Taille Couche Cachée 1 :",
        min_value=4,
        max_value=64,
        value=16,
        step=4
    )
    
    hidden_layer_2 = st.sidebar.slider(
        "Taille Couche Cachée 2 :",
        min_value=0,
        max_value=32,
        value=8,
        step=4,
        help="Mettez à 0 pour une seule couche cachée"
    )
    
    activation = st.sidebar.selectbox(
        "Fonction d'Activation :",
        ["relu", "tanh", "sigmoid"]
    )
    
    learning_rate = st.sidebar.slider(
        "Taux d'Apprentissage :",
        min_value=0.001,
        max_value=0.5,
        value=0.01,
        step=0.001,
        format="%.3f"
    )
    
    epochs = st.sidebar.slider(
        "Époques d'Entraînement :",
        min_value=100,
        max_value=2000,
        value=500,
        step=100
    )
    
    batch_size = st.sidebar.selectbox(
        "Taille du Lot :",
        [8, 16, 32, 64],
        index=1
    )
    
    if st.sidebar.button("Entraîner le Modèle", type="primary"):
        source = 'database' if 'Base de données' in data_source else 'sklearn'
        X_train, X_test, y_train, y_test, feature_names, class_names = load_iris_dataset(source=source)
        
        if hidden_layer_2 > 0:
            layer_sizes = [4, hidden_layer_1, hidden_layer_2, 3]
        else:
            layer_sizes = [4, hidden_layer_1, 3]
        
        mlp = MLP(
            layer_sizes=layer_sizes,
            learning_rate=learning_rate,
            activation=activation,
            weight_init='he' if activation == 'relu' else 'xavier'
        )
        
        st.subheader("Architecture du Réseau")
        st.code(mlp.get_architecture_summary())
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with st.spinner("Entraînement en cours..."):
            history = mlp.fit(
                X_train, y_train, 
                epochs=epochs, 
                batch_size=batch_size, 
                verbose=False,
                validation_data=(X_test, y_test)
            )
        
        progress_bar.progress(100)
        status_text.text("Entraînement terminé !")
        
        y_pred_train = mlp.predict(X_train)
        y_pred_test = mlp.predict(X_test)
        y_proba_test = mlp.predict_proba(X_test)
        
        train_metrics = compute_metrics(y_train, y_pred_train, list(class_names))
        test_metrics = compute_metrics(y_test, y_pred_test, list(class_names))
        
        st.subheader("Courbes d'Entraînement")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        epochs_range = range(len(history['loss']))
        
        axes[0].plot(epochs_range, history['loss'], label='Perte Entraînement')
        if history['val_loss']:
            axes[0].plot(epochs_range, history['val_loss'], label='Perte Validation')
        axes[0].set_title('Perte par Époque')
        axes[0].set_xlabel('Époque')
        axes[0].set_ylabel('Perte Entropie Croisée')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(epochs_range, history['accuracy'], label='Précision Entraînement')
        if history['val_accuracy']:
            axes[1].plot(epochs_range, history['val_accuracy'], label='Précision Validation')
        axes[1].set_title('Précision par Époque')
        axes[1].set_xlabel('Époque')
        axes[1].set_ylabel('Précision')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        st.subheader("Performance du Modèle")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Ensemble d'Entraînement")
            st.metric("Précision", f"{train_metrics['accuracy']:.2%}")
            st.metric("Score F1 (Macro)", f"{train_metrics['f1_macro']:.4f}")
        
        with col2:
            st.markdown("### Ensemble de Test")
            st.metric("Précision", f"{test_metrics['accuracy']:.2%}")
            st.metric("Score F1 (Macro)", f"{test_metrics['f1_macro']:.4f}")
        
        st.subheader("Matrice de Confusion (Ensemble de Test)")
        
        fig, ax = plt.subplots(figsize=(8, 6))
        cm = test_metrics['confusion_matrix']
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=class_names,
               yticklabels=class_names,
               ylabel='Étiquette Vraie',
               xlabel='Étiquette Prédite',
               title='Matrice de Confusion')
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        st.subheader("Métriques par Classe (Ensemble de Test)")
        
        metrics_data = {
            'Classe': [m['class_name'] for m in test_metrics['per_class_metrics']],
            'Précision': [f"{m['precision']:.4f}" for m in test_metrics['per_class_metrics']],
            'Rappel': [f"{m['recall']:.4f}" for m in test_metrics['per_class_metrics']],
            'Score F1': [f"{m['f1_score']:.4f}" for m in test_metrics['per_class_metrics']],
            'Support': [m['support'] for m in test_metrics['per_class_metrics']]
        }
        st.dataframe(metrics_data, use_container_width=True)
        
        st.subheader("Exemples de Prédictions")
        
        np.random.seed(42)
        sample_indices = np.random.choice(len(y_test), min(5, len(y_test)), replace=False)
        
        samples_data = []
        for idx in sample_indices:
            true_label = class_names[y_test[idx]]
            pred_label = class_names[y_pred_test[idx]]
            probs = y_proba_test[idx]
            correct = "Oui" if y_test[idx] == y_pred_test[idx] else "Non"
            
            prob_str = ", ".join([f"{class_names[j]}: {probs[j]:.3f}" for j in range(len(class_names))])
            
            samples_data.append({
                'Étiquette Vraie': true_label,
                'Prédiction': pred_label,
                'Correct': correct,
                'Probabilités': prob_str
            })
        
        st.dataframe(samples_data, use_container_width=True)
        
        run_id = save_training_run(
            dataset_name='Iris',
            model_type='MLP',
            layer_sizes=layer_sizes,
            activation=activation,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            optimizer='sgd',
            regularization=None,
            final_train_accuracy=train_metrics['accuracy'],
            final_test_accuracy=test_metrics['accuracy'],
            final_train_loss=history['loss'][-1] if history['loss'] else None,
            final_test_loss=history['val_loss'][-1] if history['val_loss'] else None,
            train_loss_history=history['loss'],
            train_accuracy_history=history['accuracy'],
            val_loss_history=history['val_loss'],
            val_accuracy_history=history['val_accuracy'],
            confusion_matrix=cm,
            f1_score=test_metrics['f1_macro'],
            training_time_seconds=None
        )
        
        st.success(f"Entraînement terminé ! Résultats sauvegardés dans la base de données (Exécution #{run_id}).")


def show_mnist_training():
    st.header("Entraîner le MLP sur le Jeu de Données MNIST")
    
    st.markdown("""
    ## Le Jeu de Données MNIST
    
    Le jeu de données classique de reconnaissance de chiffres manuscrits :
    - **784 Caractéristiques** : Images en niveaux de gris 28x28 (aplaties)
    - **10 Classes** : Chiffres 0-9
    - **70 000 Échantillons** : 60 000 entraînement + 10 000 test (nous utilisons un sous-ensemble pour un entraînement plus rapide)
    
    C'est le "Hello World" de l'apprentissage profond, démontrant que notre MLP peut traiter de vraies données d'images !
    """)
    
    st.sidebar.header("Configuration MNIST")
    
    n_samples = st.sidebar.select_slider(
        "Nombre d'Échantillons :",
        options=[1000, 2000, 5000, 10000, 20000],
        value=5000,
        help="Moins d'échantillons = entraînement plus rapide, plus d'échantillons = meilleure précision"
    )
    
    hidden_layer_1 = st.sidebar.slider(
        "Taille Couche Cachée 1 :",
        min_value=32,
        max_value=256,
        value=128,
        step=32,
        key="mnist_h1"
    )
    
    hidden_layer_2 = st.sidebar.slider(
        "Taille Couche Cachée 2 :",
        min_value=0,
        max_value=128,
        value=64,
        step=32,
        help="Mettez à 0 pour une seule couche cachée",
        key="mnist_h2"
    )
    
    activation = st.sidebar.selectbox(
        "Fonction d'Activation :",
        ["relu", "tanh", "sigmoid"],
        key="mnist_activation"
    )
    
    learning_rate = st.sidebar.slider(
        "Taux d'Apprentissage :",
        min_value=0.001,
        max_value=0.5,
        value=0.1,
        step=0.001,
        format="%.3f",
        key="mnist_lr"
    )
    
    epochs = st.sidebar.slider(
        "Époques d'Entraînement :",
        min_value=10,
        max_value=100,
        value=30,
        step=5,
        key="mnist_epochs"
    )
    
    batch_size = st.sidebar.selectbox(
        "Taille du Lot :",
        [32, 64, 128, 256],
        index=1,
        key="mnist_batch"
    )
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Exemples de Chiffres MNIST")
        
        if 'mnist_samples' not in st.session_state:
            st.session_state.mnist_samples = None
            st.session_state.mnist_load_error = None
        
        if st.session_state.mnist_samples is None and st.session_state.mnist_load_error is None:
            if st.button("Charger les Exemples de Chiffres"):
                try:
                    with st.spinner("Chargement des échantillons MNIST..."):
                        X_sample, _, y_sample, _, _ = load_mnist_dataset(n_samples=1000)
                        sample_imgs, sample_labels = get_mnist_sample_images(X_sample, y_sample, n_per_class=2)
                        st.session_state.mnist_samples = (sample_imgs, sample_labels)
                        st.rerun()
                except Exception as e:
                    st.session_state.mnist_load_error = str(e)
                    st.rerun()
        
        if st.session_state.mnist_load_error:
            st.warning(f"Impossible de charger les échantillons MNIST : {st.session_state.mnist_load_error}")
            st.info("Vous pouvez toujours entraîner sur MNIST en cliquant sur le bouton 'Entraîner sur MNIST' dans la barre latérale.")
        elif st.session_state.mnist_samples:
            sample_imgs, sample_labels = st.session_state.mnist_samples
            
            fig, axes = plt.subplots(2, 10, figsize=(12, 3))
            for i in range(20):
                row = i // 10
                col = i % 10
                axes[row, col].imshow(sample_imgs[i], cmap='gray')
                axes[row, col].axis('off')
                axes[row, col].set_title(str(sample_labels[i]), fontsize=10)
            plt.suptitle('Exemples de Chiffres MNIST', fontsize=12)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        else:
            st.info("Cliquez sur le bouton ci-dessus pour charger les exemples de chiffres MNIST.")
    
    with col2:
        st.subheader("Infos sur le Jeu de Données")
        st.markdown("""
        **Taille de l'Image** : 28 x 28 pixels
        
        **Caractéristiques d'Entrée** : 784 (aplaties)
        
        **Normalisation** : 0-255 → 0-1
        
        **Classes** : 10 chiffres
        """)
    
    if st.sidebar.button("Entraîner sur MNIST", type="primary", key="mnist_train_btn"):
        with st.spinner(f"Chargement de {n_samples} échantillons MNIST..."):
            X_train, X_test, y_train, y_test, class_names = load_mnist_dataset(n_samples=n_samples)
        
        st.info(f"{len(X_train)} échantillons d'entraînement et {len(X_test)} échantillons de test chargés")
        
        if hidden_layer_2 > 0:
            layer_sizes = [784, hidden_layer_1, hidden_layer_2, 10]
        else:
            layer_sizes = [784, hidden_layer_1, 10]
        
        mlp = MLP(
            layer_sizes=layer_sizes,
            learning_rate=learning_rate,
            activation=activation,
            weight_init='he' if activation == 'relu' else 'xavier'
        )
        
        st.subheader("Architecture du Réseau")
        st.code(mlp.get_architecture_summary())
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Entraînement en cours... Cela peut prendre un moment pour MNIST.")
        
        with st.spinner("Entraînement du MLP sur MNIST..."):
            history = mlp.fit(
                X_train, y_train, 
                epochs=epochs, 
                batch_size=batch_size, 
                verbose=False,
                validation_data=(X_test, y_test)
            )
        
        progress_bar.progress(100)
        status_text.text("Entraînement terminé !")
        
        y_pred_train = mlp.predict(X_train)
        y_pred_test = mlp.predict(X_test)
        y_proba_test = mlp.predict_proba(X_test)
        
        train_metrics = compute_metrics(y_train, y_pred_train, class_names)
        test_metrics = compute_metrics(y_test, y_pred_test, class_names)
        
        st.subheader("Courbes d'Entraînement")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        epochs_range = range(len(history['loss']))
        
        axes[0].plot(epochs_range, history['loss'], label='Perte Entraînement')
        if history['val_loss']:
            axes[0].plot(epochs_range, history['val_loss'], label='Perte Validation')
        axes[0].set_title('Perte par Époque')
        axes[0].set_xlabel('Époque')
        axes[0].set_ylabel('Perte Entropie Croisée')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(epochs_range, history['accuracy'], label='Précision Entraînement')
        if history['val_accuracy']:
            axes[1].plot(epochs_range, history['val_accuracy'], label='Précision Validation')
        axes[1].set_title('Précision par Époque')
        axes[1].set_xlabel('Époque')
        axes[1].set_ylabel('Précision')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        st.subheader("Performance du Modèle")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Ensemble d'Entraînement")
            st.metric("Précision", f"{train_metrics['accuracy']:.2%}")
            st.metric("Score F1 (Macro)", f"{train_metrics['f1_macro']:.4f}")
        
        with col2:
            st.markdown("### Ensemble de Test")
            st.metric("Précision", f"{test_metrics['accuracy']:.2%}")
            st.metric("Score F1 (Macro)", f"{test_metrics['f1_macro']:.4f}")
        
        st.subheader("Matrice de Confusion (Ensemble de Test)")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        cm = test_metrics['confusion_matrix']
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=class_names,
               yticklabels=class_names,
               ylabel='Étiquette Vraie',
               xlabel='Étiquette Prédite',
               title='Matrice de Confusion MNIST')
        
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=8)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        st.subheader("Exemples de Prédictions")
        
        np.random.seed(42)
        sample_indices = np.random.choice(len(y_test), min(10, len(y_test)), replace=False)
        
        fig, axes = plt.subplots(2, 5, figsize=(12, 5))
        for i, idx in enumerate(sample_indices):
            row = i // 5
            col = i % 5
            img = X_test[idx].reshape(28, 28)
            true_label = y_test[idx]
            pred_label = y_pred_test[idx]
            correct = true_label == pred_label
            
            axes[row, col].imshow(img, cmap='gray')
            axes[row, col].axis('off')
            color = 'green' if correct else 'red'
            axes[row, col].set_title(f'Vrai : {true_label}, Préd : {pred_label}', 
                                      color=color, fontsize=10)
        
        plt.suptitle('Exemples de Prédictions (Vert=Correct, Rouge=Faux)', fontsize=12)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        run_id = save_training_run(
            dataset_name='MNIST',
            model_type='MLP',
            layer_sizes=layer_sizes,
            activation=activation,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            optimizer='sgd',
            regularization=None,
            final_train_accuracy=train_metrics['accuracy'],
            final_test_accuracy=test_metrics['accuracy'],
            final_train_loss=history['loss'][-1] if history['loss'] else None,
            final_test_loss=history['val_loss'][-1] if history['val_loss'] else None,
            train_loss_history=history['loss'],
            train_accuracy_history=history['accuracy'],
            val_loss_history=history['val_loss'],
            val_accuracy_history=history['val_accuracy'],
            confusion_matrix=cm,
            f1_score=test_metrics['f1_macro'],
            training_time_seconds=None
        )
        
        st.success(f"Entraînement terminé ! Résultats sauvegardés dans la base de données (Exécution #{run_id}).")
        st.balloons()


def show_training_history():
    st.header("Historique d'Entraînement")
    
    st.markdown("""
    Visualisez et analysez vos exécutions d'entraînement précédentes stockées dans la base de données.
    Comparez différentes configurations de modèles et leurs performances.
    """)
    
    history = get_training_history(limit=20)
    
    if not history:
        st.info("Aucune exécution d'entraînement trouvée. Entraînez un modèle sur le jeu de données Iris pour voir votre historique ici.")
        return
    
    st.subheader(f"Exécutions d'Entraînement Récentes ({len(history)} au total)")
    
    history_df = []
    for run in history:
        history_df.append({
            'ID': run['id'],
            'Horodatage': run['timestamp'][:19].replace('T', ' '),
            'Jeu de Données': run['dataset_name'],
            'Architecture': ' → '.join(map(str, run['layer_sizes'])),
            'Activation': run['activation'],
            'TA': f"{run['learning_rate']:.4f}",
            'Époques': run['epochs'],
            'Préc. Entr.': f"{run['final_train_accuracy']:.2%}" if run['final_train_accuracy'] else 'N/A',
            'Préc. Test': f"{run['final_test_accuracy']:.2%}" if run['final_test_accuracy'] else 'N/A',
            'F1': f"{run['f1_score']:.4f}" if run['f1_score'] else 'N/A'
        })
    
    st.dataframe(history_df, use_container_width=True)
    
    st.subheader("Voir les Détails de l'Exécution")
    
    run_ids = [run['id'] for run in history]
    selected_run_id = st.selectbox(
        "Sélectionner une exécution d'entraînement pour voir les détails :",
        options=run_ids,
        format_func=lambda x: f"Exécution #{x} - {next((r['timestamp'][:19].replace('T', ' ') for r in history if r['id'] == x), '')}"
    )
    
    if selected_run_id:
        run_details = get_training_run_details(selected_run_id)
        
        if run_details:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Configuration du Modèle")
                st.write(f"**Jeu de Données :** {run_details['dataset_name']}")
                st.write(f"**Modèle :** {run_details['model_type']}")
                st.write(f"**Architecture :** {' → '.join(map(str, run_details['layer_sizes']))}")
                st.write(f"**Activation :** {run_details['activation']}")
                st.write(f"**Taux d'Apprentissage :** {run_details['learning_rate']}")
                st.write(f"**Époques :** {run_details['epochs']}")
                st.write(f"**Taille du Lot :** {run_details['batch_size']}")
            
            with col2:
                st.markdown("### Métriques de Performance")
                if run_details['final_train_accuracy']:
                    st.metric("Précision Entraînement", f"{run_details['final_train_accuracy']:.2%}")
                if run_details['final_test_accuracy']:
                    st.metric("Précision Test", f"{run_details['final_test_accuracy']:.2%}")
                if run_details['f1_score']:
                    st.metric("Score F1", f"{run_details['f1_score']:.4f}")
            
            if run_details['train_loss_history'] and run_details['train_accuracy_history']:
                st.markdown("### Courbes d'Entraînement")
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                
                epochs_range = range(len(run_details['train_loss_history']))
                
                axes[0].plot(epochs_range, run_details['train_loss_history'], label='Perte Entraînement')
                if run_details['val_loss_history']:
                    axes[0].plot(epochs_range, run_details['val_loss_history'], label='Perte Validation')
                axes[0].set_title('Perte par Époque')
                axes[0].set_xlabel('Époque')
                axes[0].set_ylabel('Perte')
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)
                
                axes[1].plot(epochs_range, run_details['train_accuracy_history'], label='Précision Entraînement')
                if run_details['val_accuracy_history']:
                    axes[1].plot(epochs_range, run_details['val_accuracy_history'], label='Précision Validation')
                axes[1].set_title('Précision par Époque')
                axes[1].set_xlabel('Époque')
                axes[1].set_ylabel('Précision')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            
            if run_details['confusion_matrix'] is not None:
                st.markdown("### Matrice de Confusion")
                cm = run_details['confusion_matrix']
                class_names = ['setosa', 'versicolor', 'virginica']
                
                fig, ax = plt.subplots(figsize=(6, 5))
                im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                ax.figure.colorbar(im, ax=ax)
                
                ax.set(xticks=np.arange(cm.shape[1]),
                       yticks=np.arange(cm.shape[0]),
                       xticklabels=class_names,
                       yticklabels=class_names,
                       ylabel='Étiquette Vraie',
                       xlabel='Étiquette Prédite')
                
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
                
                thresh = cm.max() / 2.
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        ax.text(j, i, format(int(cm[i, j]), 'd'),
                                ha="center", va="center",
                                color="white" if cm[i, j] > thresh else "black")
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            
            if st.button("Supprimer Cette Exécution", type="secondary"):
                if delete_training_run(selected_run_id):
                    st.success(f"Exécution #{selected_run_id} supprimée avec succès.")
                    st.rerun()
                else:
                    st.error("Échec de la suppression de l'exécution.")


if __name__ == "__main__":
    main()

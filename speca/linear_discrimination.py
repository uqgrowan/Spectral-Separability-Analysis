import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import f1_score, confusion_matrix

# =============================================================================
# Step 6: Linear Discriminant Analysis (LDA)
# =============================================================================
def LDA_classify (spectra, labels):
    # Split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        spectra, labels, test_size=0.3, stratify=labels)

    # Perform LDA
    lda = LinearDiscriminantAnalysis(solver = "svd")#"eigen", shrinkage = "auto")
    lda.fit(X_train, y_train)
    predictions = lda.predict(X_test)

    results = {}
    # Predict on test set
    results["pred"] = predictions
    results["accuracy"] = lda.score(X_test, y_test)
    results["f1_score"] = f1_score(y_test, predictions, average = "macro")
    cm = confusion_matrix(y_test, predictions, normalize = 'true')
    results["cm"] = cm

    return results

# =============================================================================
# Step 7: Plot LDA components
# =============================================================================


def LDA_transform(spectra, labels, test_size =0.2):
    # Split data for training and testing
    X_train, _, y_train, _ = train_test_split(
        spectra, labels, test_size=test_size,random_state=100, stratify=labels)

    # Perform LDA
    lda = LinearDiscriminantAnalysis(solver = "eigen", shrinkage = "auto")
    lda.fit(X_train, y_train)
    
    # Transform data to LDA space
    X_lda = lda.transform(spectra)
    
    return X_lda

def LDA_plot_comps(X_lda, labels, components = [0,1,2]):

    unique_categories = np.unique(labels)
    
    
    if len(components) == 3:
        
        # Plot three LDA components
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(projection='3d')
        x, y, z = components
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_categories)))

        for i, category in enumerate(unique_categories):
            mask = labels == category
            if X_lda.shape[1] >= 2:
                ax.scatter(X_lda[mask, x], X_lda[mask, y], X_lda[mask, z],
                        c=[colors[i]], label=category, alpha=0.7, s=50)
            else:
                ax.scatter(X_lda[mask, 0], np.zeros(np.sum(mask)), 
                        c=[colors[i]], label=category, alpha=0.7, s=50)

    elif len(components) == 2:
        
        # Plot two LDA components
        plt.figure(figsize=(14, 10))
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_categories)))
        x, y = components
        for i, category in enumerate(unique_categories):
            mask = labels == category
            if X_lda.shape[1] >= 2:
                plt.scatter(X_lda[mask, x], 
                           X_lda[mask, y],
                           c=[colors[i]],
                           label=category,
                           alpha=0.7,
                           s=50)
            else:
                plt.scatter(X_lda[mask, 0],
                           np.zeros(np.sum(mask)),
                           c=[colors[i]],
                           label=category,
                           alpha=0.7, s=50)
    else:
        raise ValueError("either 2 or 3 components must be specified.")
    #ax.yaxis.set_inverted(True)
    plt.xlabel('First Discriminant Component')
    plt.ylabel('Second Discriminant Component' if X_lda.shape[1] >= 2 else '0')
    if len(components) == 3:
        ax.set_zlabel('Third Discriminant Component')
    plt.title('Linear Discriminant Analysis - Category Separation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()



def LDA_plot_cm(cm, labels):
    unique_categories = np.unique(labels)

    # Confusion matrix
    cm_df= pd.DataFrame(cm)
    cm_masked = cm_df.map(lambda v: str(int(v*100)) if int(v*100) >0 else "")
    plt.figure(figsize=(14, 10))
    sns.heatmap(cm,
                annot= cm_masked,
                cmap='Blues',
                fmt = 's',
                xticklabels=unique_categories,
                yticklabels=unique_categories)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

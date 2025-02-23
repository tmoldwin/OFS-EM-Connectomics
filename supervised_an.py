import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from metric_calc import ALL_FEATURES, FEATURE_NAMES

def load_and_prepare_data():
    """Load data and prepare features/labels for E/I classification"""
    df = pd.read_csv('data/synapse_data_with_metrics.csv')
    
    # Print class distribution
    print("\nClass distribution:")
    print(df['pre_syn_clf_type'].value_counts())
    
    # Use standardized feature names from metric_calc
    feature_cols = [col for col in df.columns if col in ALL_FEATURES]
    
    # Print available feature groups
    print("\nFeature groups used:")
    for group, features in FEATURE_NAMES.items():
        available = [f for f in features if f in df.columns]
        if available:
            print(f"\n{group.title()}:")
            print(f"Using {len(available)} features: {', '.join(available)}")
    
    # Create E/I labels
    df['is_excitatory'] = df['pre_syn_clf_type'].str.contains('E', case=False)
    y = df['is_excitatory']
    
    # Get feature importance and select top features
    X = df[feature_cols]
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X, y)
    
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 20 most important features:")
    print(importance.head(20))
    
    # Use top features
    top_features = importance.head(20)['feature'].values
    X = X[top_features].copy()
    
    # Feature engineering on top features
    for i in range(min(5, len(top_features))):
        for j in range(i+1, min(5, len(top_features))):
            ratio_name = f"{top_features[i]}_to_{top_features[j]}"
            X[ratio_name] = X[top_features[i]] / X[top_features[j]]
    
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return X, y, X.columns

def evaluate_classifiers(X, y):
    """Evaluate classifiers on full dataset with balanced training"""
    classifiers = {
        'Random Forest': RandomForestClassifier(random_state=42, 
                                              n_estimators=500,
                                              max_depth=10,
                                              min_samples_leaf=5),
        'SVM': SVC(random_state=42, 
                   kernel='rbf',
                   C=10,
                   gamma='scale',
                   probability=True),
        'Neural Net': MLPClassifier(random_state=42,
                                  hidden_layer_sizes=(50, 25),
                                  max_iter=5000,
                                  learning_rate_init=0.001,
                                  early_stopping=True)
    }
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_array = np.array(y)
    
    results = {}
    confusion_matrices = {}
    
    # Create balanced training set
    excitatory_mask = (y_array == 1)
    inhibitory_mask = ~excitatory_mask
    
    excitatory_indices = np.where(excitatory_mask)[0]
    inhibitory_indices = np.where(inhibitory_mask)[0]
    
    n_inhibitory = len(inhibitory_indices)
    balanced_excitatory_indices = np.random.choice(
        excitatory_indices, 
        size=n_inhibitory, 
        replace=False
    )
    
    balanced_indices = np.concatenate([inhibitory_indices, balanced_excitatory_indices])
    np.random.shuffle(balanced_indices)
    
    X_train_balanced = X_scaled[balanced_indices]
    y_train_balanced = y_array[balanced_indices]
    
    for name, clf in classifiers.items():
        print(f"\n{name}:")
        print("Training set balance:")
        unique, counts = np.unique(y_train_balanced, return_counts=True)
        print(dict(zip(unique, counts)))
        
        # Train on balanced data
        clf.fit(X_train_balanced, y_train_balanced)
        
        # Predict on full dataset
        if hasattr(clf, "predict_proba"):
            y_prob = clf.predict_proba(X_scaled)
            y_pred = (y_prob[:, 1] > 0.5).astype(int)
        else:
            y_pred = clf.predict(X_scaled)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_array, y_pred)
        
        # Convert to fractions
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        confusion_matrices[name] = cm_normalized
        
        # Store accuracy
        results[name] = balanced_accuracy_score(y_array, y_pred)
        
        print("\nConfusion Matrix (fractions):")
        print(pd.DataFrame(cm_normalized, 
                         index=['True I', 'True E'],
                         columns=['Pred I', 'Pred E']).round(3))
        print("\nClassification Report:")
        print(classification_report(y_array, y_pred))
    
    return results, confusion_matrices

def plot_results(results, confusion_matrices):
    """Plot confusion matrices as fractions"""
    n_classifiers = len(confusion_matrices)
    fig, axes = plt.subplots(1, n_classifiers, figsize=(15, 5))
    
    # Plot confusion matrices
    labels = ['I', 'E']
    for idx, (name, cm) in enumerate(confusion_matrices.items()):
        sns.heatmap(cm, annot=True, fmt='.3f', cmap='Blues', ax=axes[idx],
                   xticklabels=labels, yticklabels=labels, vmin=0, vmax=1)
        axes[idx].set_title(f'{name}\nAccuracy: {results[name]:.3f}')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('True')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Load and prepare data
    X, y, feature_cols = load_and_prepare_data()
    
    # Evaluate classifiers
    results, confusion_matrices = evaluate_classifiers(X, y)
    
    # Plot results
    plot_results(results, confusion_matrices)
    plt.savefig('figs/supervised.png', bbox_inches='tight', dpi=300)
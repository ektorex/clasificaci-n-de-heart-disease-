from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, f1_score
from scipy.stats import randint, uniform

param_dist = {

    # PCA
    'pca__n_components': [0.90, 0.95, 0.99],

    # Arquitectura de la red
    'modelo__hidden_layer_sizes': [
        (32,),
        (64,),
        (128,),
        (64, 32),
        (128, 64),
        (128, 64, 32)
    ],

    # Función de activación
    'modelo__activation': ['relu', 'tanh'],

    # Optimizador
    'modelo__solver': ['adam', 'sgd'],

    # Learning rate
    'modelo__learning_rate_init': uniform(0.0001, 0.01),

    # Regularización
    'modelo__alpha': uniform(0.0001, 0.01),

    # Batch size
    'modelo__batch_size': [16, 32, 64, 128]
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
scorer = make_scorer(f1_score, average='weighted')

random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_dist,
    n_iter=30,
    cv=cv,
    verbose=2,
    random_state=42,
    n_jobs=-1,
    scoring=scorer
)
##############################################################################

mpl = ('modelo', MLPClassifier(
        activation='tanh',
        alpha=0.00373629602379294,
        batch_size=16,
        beta_1=0.9,
        beta_2=0.999,
        early_stopping=True,
        hidden_layer_sizes=(64,),
        max_iter=2000,
        learning_rate_init=0.009724472949421113,
        solver='sgd',
        random_state=42
    ))

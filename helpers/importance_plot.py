import numpy as np
import matplotlib.pyplot as plt


def importance_plot(features, feature_importance):
    sorted_idx = np.argsort(feature_importance)
    sorted_feats = []
    for i in sorted_idx:
        sorted_feats.append(features[i])

    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.figure(figsize=(20, 20))
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, sorted_feats)
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    plt.show()

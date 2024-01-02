# %% Imports & Constant
import numpy as np
from sklearn.linear_model import RidgeCV
from rrrr import ReducedRankRidgeRegressionCV

Alphas = np.logspace(-3, 8, 10)
nSamples, nVoxels = 500, 1000

## Useful Methods
def _simulate_XY(r, nSamples, nVoxels, nRank, std=.01):
    '''
        Create the (nSamples) for input data (X) and target (Y)
        following :
            Y = XB + E s.t. Y is low-rank (nRank) with spatially
            correlated voxels (nVodels) with strength (r)

        Note : 
            - To create the Data follow proposed Simulation Model 1 from :
            Reduced Rank Ridge Regression following the paper from [Mukherjee2011]
            https://dept.stat.lsa.umich.edu/~jizhu/pubs/Mukherjee-SADM11.pdf
    '''

    ## Step 1 : Spatially correlated X (signal) drawn from N(0,S)
    S = np.zeros((nRank, nRank))
    for i in range(nRank):
        for j in range(nRank):
            S[i,j] = r**np.abs(i-j)

    X = np.random.multivariate_normal(
        mean=np.zeros(nRank),
        cov=S,
        size=nSamples
    )

    ## Step 2 : Create B (hidden linear projection between X and Y)
    B = np.random.multivariate_normal(
        mean=np.zeros(nVoxels),
        cov=np.identity(nVoxels),
        size=nRank
    )

    ## Model 1 : Replace half of B's singular values of s by 2, rest set at 0
    u, s, vh = np.linalg.svd(B, full_matrices=False)
    s[:len(s)//2] = 2
    s[len(s)//2:] = 0
    B = u @ np.diag(s) @ vh

    ## Step 3 : Create Error E from N(0,1)
    E = np.random.multivariate_normal(
        mean=np.zeros(nVoxels),
        cov= std * np.identity(nVoxels),
        size=nSamples
    )

    ## 
    return X, X @ B + E

# %% Execute
R = [.001, .3, .5, .7, .9]; nR = len(R)
Ranks = [i for i in range(10, 210, 10)]; nRanks = len(Ranks)

rrrr  = ReducedRankRidgeRegressionCV(Alphas, Ranks)
ridge = RidgeCV(Alphas)

Estimated_Rank, Scores_RRRR, Scores_Ridge = np.zeros((nRanks, nR)), np.zeros((nRanks, nR)), np.zeros((nRanks, nR))
for i in range(nRanks):
    for j in range(nR):

        ## Create Train/Test Sets (from same distribution)
        X, Y = _simulate_XY(R[j], nSamples, nVoxels, Ranks[i])
        X_train, X_test = X[:int(.6*nSamples)], X[int(.6*nSamples):]
        Y_train, Y_test = Y[:int(.6*nSamples)], Y[int(.6*nSamples):]

        ## Fit the Regression Models on Train Set
        rrrr.fit(X_train,  Y_train)
        ridge.fit(X_train, Y_train)

        ## Save the estimated rank
        Estimated_Rank[i,j] = rrrr.rank_

        ## Assess the Model's performances on Test Set
        Scores_RRRR[i,j]  = rrrr.score(X_test,  Y_test)
        Scores_Ridge[i,j] = ridge.score(X_test, Y_test)

# %% Create Summary Figures
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

Colors  = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
Legend  = [Line2D([0], [0], marker='o', color='w', label=rf'$\rho={R[j]}$', markerfacecolor=Colors[j], markeredgecolor='w', markersize=9) for j in range(nR)]

## Estimated Rank by RRRR
plt.figure()
for i in range(nRanks):
    for j in range(nR):
        plt.plot([Ranks[i]], [Estimated_Rank[i, j]], '.', color=Colors[j])

plt.plot([min(Ranks), max(Ranks)], [min(Ranks), max(Ranks)], 'k--')

plt.xlabel('True Hidden Model Rank', fontsize=13)
plt.ylabel('Estimated Rank', fontsize=13)
plt.legend(
    handles=Legend,
    title='spatial corr strength',
    title_fontsize=10)
plt.title('Y=XB+E / Y spatially correlated with low-rank', fontsize=13)

import colorsys
import matplotlib.colors

def colors_gradient(Colors, luminance_levels):
    '''
        Transform a list of RGB colors (defined in Hex)
        to a list of list of RGB Colors with every sublist containing
        the gradient of previous base-color for different luminance level (in hls space)
    '''
    Colors_Gradients = []
    for color in Colors:
        r,g,b = matplotlib.colors.to_rgb(color)
        h,l,s = colorsys.rgb_to_hls(r,g,b)
        gradient = []
        for luminance_level in luminance_levels:
            gradient.append(colorsys.hls_to_rgb(h, luminance_level, s))
        Colors_Gradients.append(gradient.copy())
    return Colors_Gradients

Colors = ['#0094FF']

luminance_levels = np.linspace(.9, .1, nRanks)[::-1]
Colors_Gradient = colors_gradient(Colors, luminance_levels)[0]

## Performances Accuracy (R2-score) Ridge vs. RRRR
Scores = np.concatenate((Scores_Ridge, Scores_RRRR))
score_min, score_max = np.min(Scores), np.max(Scores)

Legend  = [Line2D([0], [0], marker='o', color='w', label='Small Rank', markerfacecolor=Colors_Gradient[0], markeredgecolor='w', markersize=9),
           Line2D([0], [0], marker='o', color='w', label='High Rank', markerfacecolor=Colors_Gradient[-1], markeredgecolor='w', markersize=9)]

plt.figure()
for i in range(nRanks):
    plt.plot(Scores_Ridge[i,:], Scores_RRRR[i,:], '.', color=Colors_Gradient[i])
plt.plot([score_min, score_max], [score_min, score_max], 'k--')

plt.xlabel(r'Ridge CV Score [$r^2$]', fontsize=14)
plt.ylabel(r'RRRR CV Score [$r^2$]', fontsize=14)
plt.title('Y=XB+E / Y spatially correlated with low-rank', fontsize=13)
plt.legend(handles=Legend, fontsize=13)

# %%

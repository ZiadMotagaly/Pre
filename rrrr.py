from sklearn.linear_model import RidgeCV
from sklearn.decomposition import TruncatedSVD
from sklearn import base, metrics
import numpy as np

class ReducedRankRidgeRegressionCV(base.BaseEstimator):
    """
    scikit-learn like estimator for the RRRR with built-in Cross-Validation

    Attributes
    ----------

    coef_ : ndarray of shape (n_features, n_targets)
    intercept_ : ndarray of shape (n_targets,)
        Estimated Coefficients & Intercept for the Reduced Rank Ridge Regression

    alpha_ : ndarray of shape (n_targets,)
        Estimated regularization parameter for each target for the Ridge Regression

    rank_ : int
        Estimated Optimal Reduced Rank for the Projection of the Ridge Prediction

    best_score_ : float
        Optimal R2 score obtained on the inner cross-validation loops on 25% of the data

    """

    def __init__(self, alphas, ranks):
        self.alphas = alphas
        self.ranks = ranks

    def fit(self, X, Y):
        """
        Reduced Rank Ridge Regression following the paper from [Mukherjee2011]
        https://dept.stat.lsa.umich.edu/~jizhu/pubs/Mukherjee-SADM11.pdf

        Idea : 
            - Step 1: using RidgeCV to obtain the optimal ridge hyperparameter penalisation
            - Step 2: gridsearch for the rank hyperameters, using the prediction of the optimal ridge
            - Step 3: computing the optimal reduced rank coefficients from alpha_opt & rank_opt

        Parameters
        ----------

        X : array, shape (n, p)
        Y : array, shape (n, q)

        """
        ridge = RidgeCV(alphas=self.alphas, alpha_per_target=True, fit_intercept=True)

        ## Step 1 : Finding best Ridge hyperparameter using the closed-form formula of RidgeCV
        ## Remark : The implicit hypothesis for using this is that all samples are independent !

        ridge.fit(X, Y)
        self.alpha_ = ridge.alpha_

        ## Splitting Data into train/validation sets for the rank grid search (75/25)
        n = len(X)
        Xt, Xv, Yv = X[:int(.75*n)], X[int(.75*n):], Y[int(.75*n):]
        Yt = ridge.predict(Xt)
        

        ## Step 2 : Finding best rank hyperparameter from the projection of ridge Predition to low-rank space
        score = []
        for rank in self.ranks:

            svd = TruncatedSVD(n_components=rank)
            svd.fit(Yt)
            Vr = svd.components_.T # shape: (q, rank)

            ## validation score - using correlation.
            Yv_pred = ridge.predict(Xv) @ Vr @ Vr.T
            score.append(metrics.r2_score(Yv.ravel(), Yv_pred.ravel()))

        idx_opt = np.argmax(score)
        self.rank_ = self.ranks[idx_opt]
        self.best_score_ = score[idx_opt]

        ## Step 3 : Computing the optimal coefficent for the RRRR on full data
        svd = TruncatedSVD(n_components=self.rank_)
        svd.fit(ridge.predict(X))
        Vr = svd.components_.T
        self.coef_ = ridge.coef_.T @ Vr @ Vr.T # shape: (p, q)
        self.intercept_ = ridge.intercept_ # shape: (q)

    def predict(self, X):
        return self.intercept_ + X @ self.coef_

    def score(self, X, y):
        y_pred = self.predict(X)
        return metrics.r2_score(y.ravel(), y_pred.ravel())

    def eval(self, y_pred, y_true):
        return self.corr(y_pred, y_true)

    def corr(self, X, Y, axis=0):
        """
        Compute the pearson correlation between X and Y, handling the constant columns.

        Remark :
            All NaN values coming from constant columns are replaced by zeros.

        Parameters
        ----------
        X, Y: array, shape (n, p)

        """
        mX = X - np.mean(X, axis=axis, keepdims=True)
        mY = Y - np.mean(Y, axis=axis, keepdims=True)
        norm_mX = np.sqrt(np.sum(mX**2, axis=axis, keepdims=True))
        norm_mX[norm_mX == 0] = 1.0
        norm_mY = np.sqrt(np.sum(mY**2, axis=axis, keepdims=True))
        norm_mY[norm_mY == 0] = 1.0

        return np.sum(mX / norm_mX * mY / norm_mY, axis=axis)
# Author:
#          Scott Lowe
#
# Based on rfe.py by authors:
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Vincent Michel <vincent.michel@inria.fr>
#          Gilles Louppe <g.louppe@gmail.com>
#
# License: BSD 3 clause

"""Recursive feature addition for feature ranking"""

import warnings
import numpy as np
from sklearn.utils import check_X_y, safe_sqr
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.base import BaseEstimator
from sklearn.base import MetaEstimatorMixin
from sklearn.base import clone
from sklearn.base import is_classifier
from sklearn.cross_validation import _check_cv as check_cv
from sklearn.cross_validation import _safe_split, _score
from sklearn.metrics.scorer import check_scoring
from sklearn.feature_selection.base import SelectorMixin


class RFA(BaseEstimator, MetaEstimatorMixin, SelectorMixin):
    """Feature ranking with recursive feature addition.
    Given an external estimator and an objective function, the goal of recursive
    feature addition (RFA) is to select features by trying all features on their
    own, selecting the best, then trying all the other features with this
    feature and selecting the best of these. This process is recursively
    repeated until enough features are added so that adding any more will 
    decrease performance.
    Parameters
    ----------
    estimator : object
        A supervised learning estimator with a `fit` method.
    n_features_to_select : int or None (default=None)
        The number of features to select. If `None`, the feature selector haults
        only when no additional features will help improve performance.
    step : int or float, optional (default=1)
        If greater than or equal to 1, then `step` corresponds to the (integer)
        number of features to add at each iteration.
        If within (0.0, 1.0), then `step` corresponds to the percentage
        (rounded down) of features to add at each iteration.
    estimator_params : dict
        Parameters for the external estimator.
        Useful for doing grid searches when an `RFA` object is passed as an
        argument to, e.g., a `sklearn.grid_search.GridSearchCV` object.
    verbose : int, default=0
        Controls verbosity of output.
    Attributes
    ----------
    n_features_ : int
        The number of selected features.
    support_ : array of shape [n_features]
        The mask of selected features.
    ranking_ : array of shape [n_features]
        The feature ranking, such that ``ranking_[i]`` corresponds to the
        ranking position of the i-th feature. Unselected (i.e., estimated
        worst) features are assigned rank ``n_features_``.
    estimator_ : object
        The external estimator fit on the reduced dataset.
    Examples
    --------
    The following example shows how to retrieve the 5 right informative
    features in the Friedman #1 dataset.
    >>> from sklearn.datasets import make_friedman1
    >>> import RFA
    >>> from sklearn.svm import SVR
    >>> X, y = make_friedman1(n_samples=50, n_features=10, random_state=0)
    >>> estimator = SVR(kernel="linear")
    >>> selector = RFA(estimator, 5, step=1)
    >>> selector = selector.fit(X, y)
    >>> selector.support_
        [some output here]
    >>> selector.ranking_
        [some output here]
    """
    def __init__(self, estimator, n_features_to_select=None, step=1,
                 estimator_params=None, verbose=0):
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.step = step
        self.estimator_params = estimator_params
        self.verbose = verbose

    def fit(self, X, y):
        """Fit the RFA model and then the underlying estimator on the selected
           features.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            The training input samples.
        y : array-like, shape = [n_samples]
            The target values.
        """
        X, y = check_X_y(X, y, "csc")
        # Initialization
        n_features = X.shape[1]
        if self.n_features_to_select is None:
            n_features_to_select = n_features
        else:
            n_features_to_select = self.n_features_to_select

        if 0.0 < self.step < 1.0:
            step = int(max(1, self.step * n_features))
        else:
            step = int(self.step)
        if step <= 0:
            raise ValueError("Step must be >0")

        if self.estimator_params is not None:
            warnings.warn("The parameter 'estimator_params' is deprecated as of version 0.16 "
                          "and will be removed in 0.18. The parameter is no longer "
                          "necessary because the value is set via the estimator initialisation "
                          "or set_params function."
                          , DeprecationWarning)

        support_ = np.zeros(n_features, dtype=np.bool)
        ranking_ = n_features * np.ones(n_features, dtype=np.int)
        last_score = None
        # Feature addition
        while np.sum(support_) < n_features_to_select:
            # Previously added features
            features_already = np.arange(n_features)[support_]
            # Features to test
            features_to_test = np.arange(n_features)[!support_]

            # Rank the remaining features
            estimator = clone(self.estimator)
            if self.estimator_params:
                estimator.set_params(**self.estimator_params)
            if self.verbose > 0:
                print("Fitting estimator with %d features." % np.sum(support_)+1)
            
            scores = np.zeros(len(features_to_test))
            for feature_index, test_feature in enumerate(features_to_test):
                estimator.fit(X[:, features], y)
                scores[feature_index] = TEST_FUNCTION(estimator)
            
            # Sort the scores in ascending order
            score_order_index = np.argsort(scores)
            ordered_scores   = scores[score_order_index]
            ordered_features = features_to_test[score_order_index]
            
            # Break if no features can improve score
            if last_score < ordered_scores[0]:
                break
            
            # Only add `step` many features if it doesn't take us past the target
            n_add = min(step, n_features_to_select - np.sum(support_))
            
            # Only add features which don't make performance go down
            n_add = min(n_add, len(np.nonzero(ordered_scores < last_score)))
            
            # Select best.
            # We will MINIMISE scoring function!!!
            features_to_add = ordered_features[0:n_add]
            
            # Add the features
            support_[features_to_add] = True
            ranking_[features_to_add] = np.sum(support_) + 1 + np.arange(features_to_add)
            
            # Update score monitor
            last_score = ordered_scores[0]
            

        # Set final attributes
        self.estimator_ = clone(self.estimator)
        if self.estimator_params:
            self.estimator_.set_params(**self.estimator_params)
        self.estimator_.fit(X[:, support_], y)
        self.n_features_ = support_.sum()
        self.support_ = support_
        self.ranking_ = ranking_

        return self

    @if_delegate_has_method(delegate='estimator')
    def predict(self, X):
        """Reduce X to the selected features and then predict using the
           underlying estimator.
        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            The input samples.
        Returns
        -------
        y : array of shape [n_samples]
            The predicted target values.
        """
        return self.estimator_.predict(self.transform(X))

    @if_delegate_has_method(delegate='estimator')
    def score(self, X, y):
        """Reduce X to the selected features and then return the score of the
           underlying estimator.
        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            The input samples.
        y : array of shape [n_samples]
            The target values.
        """
        return self.estimator_.score(self.transform(X), y)

    def _get_support_mask(self):
        return self.support_

    @if_delegate_has_method(delegate='estimator')
    def decision_function(self, X):
        return self.estimator_.decision_function(self.transform(X))

    @if_delegate_has_method(delegate='estimator')
    def predict_proba(self, X):
        return self.estimator_.predict_proba(self.transform(X))

    @if_delegate_has_method(delegate='estimator')
    def predict_log_proba(self, X):
        return self.estimator_.predict_log_proba(self.transform(X))


class RFACV(RFA, MetaEstimatorMixin):
    """Feature ranking with recursive feature elimination and cross-validated
    selection of the best number of features.
    Parameters
    ----------
    estimator : object
        A supervised learning estimator with a `fit` method.
    step : int or float, optional (default=1)
        If greater than or equal to 1, then `step` corresponds to the (integer)
        number of features to remove at each iteration.
        If within (0.0, 1.0), then `step` corresponds to the percentage
        (rounded down) of features to remove at each iteration.
    cv : int or cross-validation generator, optional (default=None)
        If int, it is the number of folds.
        If None, 3-fold cross-validation is performed by default.
        Specific cross-validation objects can also be passed, see
        `sklearn.cross_validation module` for details.
    scoring : string, callable or None, optional, default: None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.
    estimator_params : dict
        Parameters for the external estimator.
        Useful for doing grid searches when an `RFA` object is passed as an
        argument to, e.g., a `sklearn.grid_search.GridSearchCV` object.
    verbose : int, default=0
        Controls verbosity of output.
    Attributes
    ----------
    n_features_ : int
        The number of selected features with cross-validation.
    support_ : array of shape [n_features]
        The mask of selected features.
    ranking_ : array of shape [n_features]
        The feature ranking, such that `ranking_[i]`
        corresponds to the ranking
        position of the i-th feature.
        Selected (i.e., estimated best)
        features are assigned rank 1.
    grid_scores_ : array of shape [n_subsets_of_features]
        The cross-validation scores such that
        ``grid_scores_[i]`` corresponds to
        the CV score of the i-th subset of features.
    estimator_ : object
        The external estimator fit on the reduced dataset.
    Notes
    -----
    The size of ``grid_scores_`` is equal to (n_features + step - 2) // step + 1,
    where step is the number of features removed at each iteration.
    Examples
    --------
    The following example shows how to retrieve the a-priori not known 5
    informative features in the Friedman #1 dataset.
    >>> from sklearn.datasets import make_friedman1
    >>> import RFACV
    >>> from sklearn.svm import SVR
    >>> X, y = make_friedman1(n_samples=50, n_features=10, random_state=0)
    >>> estimator = SVR(kernel="linear")
    >>> selector = RFACV(estimator, step=1, cv=5)
    >>> selector = selector.fit(X, y)
    >>> selector.support_ # doctest: +NORMALIZE_WHITESPACE
    OUTPUT
    >>> selector.ranking_
    OUTPUT
    """
    def __init__(self, estimator, step=1, cv=None, scoring=None,
                 estimator_params=None, verbose=0):
        self.estimator = estimator
        self.step = step
        self.cv = cv
        self.scoring = scoring
        self.estimator_params = estimator_params
        self.verbose = verbose

    def fit(self, X, y):
        """Fit the RFA model and automatically tune the number of selected
           features.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the total number of features.
        y : array-like, shape = [n_samples]
            Target values (integers for classification, real numbers for
            regression).
        """
        X, y = check_X_y(X, y, "csr")
        if self.estimator_params is not None:
            warnings.warn("The parameter 'estimator_params' is deprecated as of version 0.16 "
                          "and will be removed in 0.18. The parameter is no longer "
                          "necessary because the value is set via the estimator initialisation "
                          "or set_params function."
                          , DeprecationWarning)
        # Initialization
        rfa = RFA(estimator=self.estimator, n_features_to_select=1,
                  step=self.step, estimator_params=self.estimator_params,
                  verbose=self.verbose - 1)

        cv = check_cv(self.cv, X, y, is_classifier(self.estimator))
        scorer = check_scoring(self.estimator, scoring=self.scoring)
        scores = np.zeros(X.shape[1])
        n_features_to_select_by_rank = np.zeros(X.shape[1])

        # Cross-validation
        for n, (train, test) in enumerate(cv):
            X_train, y_train = _safe_split(self.estimator, X, y, train)
            X_test, y_test = _safe_split(self.estimator, X, y, test, train)

            # Compute a full ranking of the features
            # ranking_ contains the same set of values for all CV folds,
            # but perhaps reordered
            ranking_ = rfa.fit(X_train, y_train).ranking_
            # Score each subset of features
            for k in range(0, np.max(ranking_)):
                indices = np.where(ranking_ <= k + 1)[0]
                estimator = clone(self.estimator)
                estimator.fit(X_train[:, indices], y_train)
                score = _score(estimator, X_test[:, indices], y_test, scorer)

                if self.verbose > 0:
                    print("Finished fold with %d / %d feature ranks, score=%f"
                          % (k + 1, np.max(ranking_), score))
                scores[k] += score
                # n_features_to_select_by_rank[k] is being overwritten
                # multiple times, but by the same value
                n_features_to_select_by_rank[k] = indices.size

        # Select the best upper bound for feature rank. It's OK to use the
        # last ranking_, as np.max(ranking_) is the same over all CV folds.
        scores = scores[:np.max(ranking_)]
        k = np.argmax(scores)

        # Re-execute an elimination with best_k over the whole set
        rfa = RFA(estimator=self.estimator,
                  n_features_to_select=n_features_to_select_by_rank[k],
                  step=self.step, estimator_params=self.estimator_params)

        rfa.fit(X, y)

        # Set final attributes
        self.support_ = rfa.support_
        self.n_features_ = rfa.n_features_
        self.ranking_ = rfa.ranking_
        self.estimator_ = clone(self.estimator)
        if self.estimator_params:
            self.estimator_.set_params(**self.estimator_params)
        self.estimator_.fit(self.transform(X), y)

        # Fixing a normalization error, n is equal to len(cv) - 1
        # here, the scores are normalized by len(cv)
        self.grid_scores_ = scores / len(cv)
        return self

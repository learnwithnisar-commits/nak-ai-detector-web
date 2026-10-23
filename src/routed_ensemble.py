"""Sklearn-compatible wrapper for v6+ domain-routed ensemble models.

The routed ensemble computes: p = R·T + (1−R)·G
where R=P(technical), T=P(AI|technical), G=P(AI|general)

This wrapper exposes the standard sklearn classifier interface
(predict_proba, predict) so the live app can use ensemble models
without code changes.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class RoutedEnsemble(BaseEstimator, ClassifierMixin):
    """Domain-routed ensemble: p(AI) = R·T + (1−R)·G."""
    
    def __init__(self, router=None, tech=None, general=None):
        self.router = router
        self.tech = tech
        self.general = general
        self.classes_ = np.array([0, 1])
    
    def predict_proba(self, X):
        r = self.router.predict_proba(X)[:, 1]
        pt = self.tech.predict_proba(X)[:, 1]
        pg = self.general.predict_proba(X)[:, 1]
        p_ai = r * pt + (1.0 - r) * pg
        p_ai = np.clip(p_ai, 0.0, 1.0)
        return np.column_stack([1.0 - p_ai, p_ai])
    
    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

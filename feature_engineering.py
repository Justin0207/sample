# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 11:25:37 2024

@author: Anyanwu Justice
"""

from sklearn.base import BaseEstimator, TransformerMixin
import scipy.stats as stats
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import numpy as np



class FeatureBinner(BaseEstimator, TransformerMixin):
    def __init__(self):
        # Initialize the bins and labels
        self.age_bins = [16, 19, 22, 26, 30, 35, 41, 49, 71]
        self.ad_grade_bins = [94.0, 107.2, 116.9, 125.2, 134.1, 144.6, 157.7, 191.0]
        self.prev_grade_bins = [94.0, 112.0, 122.0, 129.0, 136.0, 144.0, 154.0, 166.0, 191.0]
        self.labels_ad = [0, 1, 2, 3, 4, 5, 6]
        self.labels_prev = [0, 1, 2, 3, 4, 5, 6, 7]
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # Apply the binning using pd.cut
        X['age_bins'] = pd.cut(X['Age at enrollment'], bins=self.age_bins, labels=self.labels_prev).astype('int')
        
        X['Admission grade (bins)'] = pd.cut(X['Admission grade'], bins=self.ad_grade_bins, labels=self.labels_ad).astype('int')
        
        X['Previous grade (bins)'] = pd.cut(X['Previous qualification (grade)'], bins=self.prev_grade_bins, labels=self.labels_prev).astype('int')
        return X

class FeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self 
    
    def transform(self, X):
        X = X.copy()
        
        # Total Units Enrolled
        X['Total Units Enrolled'] = X['Curricular units 1st sem (enrolled)'] + X['Curricular units 2nd sem (enrolled)']
        
        # Total Units Approved
        X['Total Units Approved'] = X['Curricular units 1st sem (approved)'] + X['Curricular units 2nd sem (approved)']
        
        # Average curricular units
        X['Average curricular units'] = (X['Curricular units 1st sem (grade)'] + X['Curricular units 2nd sem (grade)']) / 2
        
        # Approval Rate (Handle division by zero to remove NaN errors)
        X['Approval Rate'] = np.where(X['Total Units Enrolled'] != 0,
                                      X['Total Units Approved'] / X['Total Units Enrolled'],
                                      0)
        
        # Improvement in Grades
        X['Improvement in Grades'] = X['Curricular units 2nd sem (grade)'] - X['Curricular units 1st sem (grade)']
        
        # Economic Hardship
        X['Economic Hardship'] = X['Unemployment rate'] + X['Inflation rate'] - X['GDP']
        
        # Total Units without Evaluations
        X['Total Units without Evaluations'] = X['Curricular units 1st sem (without evaluations)'] + X['Curricular units 2nd sem (without evaluations)']
      
        return X
class LogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = None
        pass

    def fit(self, X, y=None):
        self.columns = X.drop(['Marital status', 'Daytime/evening attendance', 'Application order', 'Nationality', 'Course', 'Application mode',
                               'Previous qualification', "Mother's qualification", "Father's qualification",
                               "Mother's occupation", "Father's occupation", 'Displaced', 'Educational special needs', 'Debtor', 'Tuition fees up to date',
                               'Gender', 'Scholarship holder', 'International', 'age_bins', 'Admission grade (bins)', 'Previous grade (bins)', 'Target'], axis = 1).columns
        return self

    def transform(self, X):
        X = X.copy()


        # Loop through specified columns
        for col in self.columns:
            # Check skewness of the column
            col_skewness = stats.skew(X[col].dropna())  # Drop NaNs to avoid skew calculation issues

            # Check if skewness is above the threshold and values are non-negative
            if -1 < col_skewness < 1 and (X[col] >= 0).all():
                X[col] = np.log1p(X[col])  # Apply log transformation
                
        return X
    
class PolynomialFeaturesInteraction(BaseEstimator, TransformerMixin):
    def __init__(self):
        # Initialize the PolynomialFeatures transformer from sklearn
        self.poly_transformer = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        
    def fit(self, X, y=None):
        # Select numerical features to apply polynomial transformation
        self.numerical_features = ['Admission grade', 'Age at enrollment', 'Curricular units 1st sem (credited)',
                                'Curricular units 1st sem (enrolled)',
                                'Curricular units 1st sem (evaluations)',
                                'Curricular units 1st sem (approved)',
                                'Curricular units 1st sem (without evaluations)',
                                'Curricular units 2nd sem (credited)',
                                'Curricular units 2nd sem (enrolled)',
                                'Curricular units 2nd sem (evaluations)',
                                'Curricular units 2nd sem (approved)',
                                'Curricular units 2nd sem (without evaluations)', 'Total Units Enrolled',
                                'Total Units Approved', 'Average curricular units', 'Approval Rate', 'Improvement in Grades',
                                'Economic Hardship', 'Total Units without Evaluations', 'Unemployment rate', 'Inflation rate', 
                                'GDP']
        # Fit the polynomial transformer on the numerical features
        self.poly_transformer.fit(X[self.numerical_features])
        return self
    
    def transform(self, X):
        # Transform the numerical features
        numerical_df = X[self.numerical_features]
        poly_features = self.poly_transformer.transform(numerical_df)
        
        # Create a DataFrame for the polynomial features
        poly_df = pd.DataFrame(poly_features, columns=self.poly_transformer.get_feature_names_out(self.numerical_features))
        
        # Drop the numerical features from the original DataFrame
        cat_df = X.drop(columns=self.numerical_features)
        
        # Concatenate the polynomial features with the categorical data
        result = pd.concat([poly_df.reset_index(drop=True), cat_df.reset_index(drop=True)], axis=1)
        return result
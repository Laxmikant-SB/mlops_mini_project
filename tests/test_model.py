# load test + signature test + performance test

import unittest
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle

class TestModelPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load the latest model from local file
        cls.model_path = './model/model.pkl'
        cls.vectorizer_path = './model/vectorizer.pkl'
        cls.test_data_path = './data/processed/test_bow.csv'

        if not os.path.exists(cls.model_path):
            raise FileNotFoundError(f"Model not found at {cls.model_path}")
        if not os.path.exists(cls.vectorizer_path):
            raise FileNotFoundError(f"Vectorizer not found at {cls.vectorizer_path}")
        if not os.path.exists(cls.test_data_path):
            raise FileNotFoundError(f"Test data not found at {cls.test_data_path}")

        with open(cls.model_path, 'rb') as f:
            cls.model = pickle.load(f)

        with open(cls.vectorizer_path, 'rb') as f:
            cls.vectorizer = pickle.load(f)

        cls.test_data = pd.read_csv(cls.test_data_path)

    def test_model_loaded(self):
        """Check if model and vectorizer are loaded properly"""
        self.assertIsNotNone(self.model)
        self.assertIsNotNone(self.vectorizer)

    def test_model_signature(self):
        """Check input-output shapes of the model"""
        # Dummy input for testing
        input_text = "hi how are you"
        input_features = self.vectorizer.transform([input_text])
        input_df = pd.DataFrame(input_features.toarray(), columns=[str(i) for i in range(input_features.shape[1])])

        # Predict
        prediction = self.model.predict(input_df)

        # Input shape should match vectorizer feature length
        self.assertEqual(input_df.shape[1], len(self.vectorizer.get_feature_names_out()))

        # Output shape
        self.assertEqual(len(prediction), input_df.shape[0])
        self.assertEqual(len(prediction.shape), 1)  # binary classification

    def test_model_performance(self):
        """Check model performance on holdout test data"""
        X_test = self.test_data.iloc[:, :-1]
        y_test = self.test_data.iloc[:, -1]

        y_pred = self.model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Define thresholds
        min_accuracy = 0.40
        min_precision = 0.40
        min_recall = 0.40
        min_f1 = 0.40

        self.assertGreaterEqual(accuracy, min_accuracy, f'Accuracy should be at least {min_accuracy}')
        self.assertGreaterEqual(precision, min_precision, f'Precision should be at least {min_precision}')
        self.assertGreaterEqual(recall, min_recall, f'Recall should be at least {min_recall}')
        self.assertGreaterEqual(f1, min_f1, f'F1 score should be at least {min_f1}')

if __name__ == "__main__":
    unittest.main()

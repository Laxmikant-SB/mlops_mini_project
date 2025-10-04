import unittest
from app import app  # Adjust the import if app.py is inside a folder

class FlaskAppTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()

    def test_home_page(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        # Check for some content that is actually in index.html
        self.assertIn(b'<form', response.data)  # Form should be present
        self.assertIn(b'name="text"', response.data)  # Text input

    def test_predict_page(self):
        response = self.client.post('/predict', data=dict(text="I love this!"))
        self.assertEqual(response.status_code, 200)
        # Since model.predict returns numeric output, just check it exists
        self.assertIn(b'result', response.data)

if __name__ == '__main__':
    unittest.main()

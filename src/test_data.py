import unittest
from datetime import datetime
from predict import predict_sales


class TestPredict(unittest.TestCase):

    def test_predict_sales(self):
        date = datetime(2026, 3, 27)
        result = predict_sales(date)
        self.assertIsInstance(result, float)


if __name__ == "__main__":
    unittest.main()
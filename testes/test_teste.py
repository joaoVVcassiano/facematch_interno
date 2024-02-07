import unittest
import cv2
import numpy as np
#add path
import sys
sys.path.append('..')
from facematch import Facematch

class TestFacematch(unittest.TestCase):
    def setUp(self):
        self.img_1 = cv2.imread("./imagens_test/biden.png")
        self.img_2 = cv2.imread("./imagens_test/istockphoto-1335941248-2048x2048.jpg")

    def test_if_image_is_not_none(self):
        """
        Test if the loaded images are not None.
        """
        self.assertIsNotNone(self.img_1)
        self.assertIsNotNone(self.img_2)

    def test_if_image_is_valid(self):
        """
        Test if the loaded images have a valid shape (3 channels).
        """
        self.assertTrue(len(self.img_1.shape) == 3)
        self.assertTrue(len(self.img_2.shape) == 3)

    def test_verify(self):
        """
        Test the 'verify' method of the Facematch class.
        """
        result = Facematch.verify(self.img_1, self.img_2)
        self.assertTrue(isinstance(result, dict))

    def test_verify_true(self):
        """
        Test the 'verify' method of the Facematch class with the same image.
        The result must be True.
        """
        result = Facematch.verify(self.img_1, self.img_1)
        self.assertTrue(result["similarity"] > 0.5)

    def test_verify_false(self):
        """
        Test the 'verify' method of the Facematch class with different images.
        The result must be False.
        """
        result = Facematch.verify(self.img_1, self.img_2)
        self.assertFalse(result["similarity"] > 0.5)

    def test_time(self):
        """
        Test the execution time of the 'verify' method of the Facematch class.
        """
        times = []
        for i in range(10):
            result = Facematch.verify(self.img_1, self.img_2)
            times.append(result["time"])
        times = times[1:]
        self.assertTrue(np.mean(times) < 1)

if __name__ == '__main__':
    unittest.main()
    
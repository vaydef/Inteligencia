import unittest
from complex_number import ComplexNumber


class TestComplexNumber(unittest.TestCase):
    def testAdd(self):
        a = ComplexNumber(3, 4)
        b = ComplexNumber(-2, 0)
        c = a + b
        self.assertEqual(c.re, 1.0)
        self.assertEqual(c.im, 4.0)

    def testSub(self):
        a = ComplexNumber(3, 4)
        b = ComplexNumber(-5, 1.2)
        c = a - b
        self.assertEqual(c.re, 8.0)
        self.assertEqual(c.im, 2.8)

    def testMul(self):
        a = ComplexNumber(4, 3)
        b = ComplexNumber(0, 1)
        c = a * b
        self.assertEqual(c.re, -3.0)
        self.assertEqual(c.im, 4.0)

    def testDiv(self):
        a = ComplexNumber(-10, -7)
        almost_one = a/a
        self.assertAlmostEqual(almost_one.re, 1.0)
        self.assertAlmostEqual(almost_one.im, 0.0)

    def testAbs(self):
        a = ComplexNumber(5, 12)
        self.assertEqual(abs(a), 13.0)

    def testInv(self):
        a = ComplexNumber(1, 1)
        b = ComplexNumber(1/2, -1/2)
        a_t = ~a
        ab = a * b
        self.assertAlmostEqual(a_t.re, b.re)
        self.assertAlmostEqual(a_t.im, b.im)
        self.assertAlmostEqual(ab.re, 1.0)
        self.assertAlmostEqual(ab.im, 0.0)

    def testRepr(self):
        a = ComplexNumber(5.1234, -3.141592)
        r = a.__repr__()
        self.assertEqual(r, '5.12 - 3.14i')

    def testReprInts(self):
        a = ComplexNumber(1, 0)
        r = a.__repr__()
        self.assertEqual(r, '1.00 + 0.00i')

    def testEqual(self):
        a = ComplexNumber(3.2, 4.5)
        b = ComplexNumber(3.2, 4.5)
        self.assertEqual(a, b)

if __name__ == '__main__':
    unittest.main()

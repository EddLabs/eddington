from eddington_test import MetaTestCase


class FitFunctionMetaTestCase(MetaTestCase):
    decimal = 5

    def test_n(self):
        self.assertEqual(self.n, self.func.n)

    def test_values(self):
        for a, x, result in self.values:
            self.assertAlmostEqual(
                result,
                self.func(a, x),
                places=self.decimal,
                msg=f"{result} != func({a}, {x})",
            )

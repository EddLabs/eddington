from unittest import TestCase

from eddington.print_util import to_relevant_precision, to_precise_string


class PrintUtilBaseTestCase:
    def test_relevant_precision(self):
        _, actual_relevant_precision = to_relevant_precision(self.a)
        self.assertEqual(
            self.relevant_precision,
            actual_relevant_precision,
            msg="Relevant precision is different than expected",
        )

    def test_precise_string(self):
        self.assertEqual(
            self.precise_string,
            to_precise_string(self.a, self.n),
            msg="Relevant precision is different than expected",
        )


class TestPrintUtilWithPositiveInteger(TestCase, PrintUtilBaseTestCase):
    a = 14
    n = 2
    relevant_precision = 0
    precise_string = "14.00"


class TestPrintUtilWithSmallInteger(TestCase, PrintUtilBaseTestCase):
    a = 3
    n = 4
    relevant_precision = 0
    precise_string = "3.0000"


class TestPrintUtilWithNegativeInteger(TestCase, PrintUtilBaseTestCase):
    a = -14
    n = 2
    relevant_precision = 0
    precise_string = "-14.00"


class TestPrintUtilWithOne(TestCase, PrintUtilBaseTestCase):
    a = 1
    n = 3
    relevant_precision = 0
    precise_string = "1.000"


class TestPrintUtilWithNegativeOne(TestCase, PrintUtilBaseTestCase):
    a = -1
    n = 3
    relevant_precision = 0
    precise_string = "-1.000"


class TestPrintUtilWithZero(TestCase, PrintUtilBaseTestCase):
    a = 0
    n = 3
    relevant_precision = 0
    precise_string = "0.000"


class TestPrintUtilWithFloatBiggerThanOneAndSmallN(TestCase, PrintUtilBaseTestCase):
    a = 3.141592
    n = 2
    relevant_precision = 0
    precise_string = "3.14"


class TestPrintUtilWithFloatBiggerThanOneAndBigN(TestCase, PrintUtilBaseTestCase):
    a = 3.52
    n = 5
    relevant_precision = 0
    precise_string = "3.52000"


class TestPrintUtilWithFloatLessThanOneAndSmallN(TestCase, PrintUtilBaseTestCase):
    a = 0.3289
    n = 1
    relevant_precision = 1
    precise_string = "0.33"


class TestPrintUtilWithFloatLessThanOneAndBigN(TestCase, PrintUtilBaseTestCase):
    a = 0.52
    n = 3
    relevant_precision = 1
    precise_string = "0.5200"


class TestPrintUtilWithVerySmallFloatLessThanOneAndSmallN(
    TestCase, PrintUtilBaseTestCase
):
    a = 3.289e-5
    n = 1
    relevant_precision = 5
    precise_string = "3.3e-05"


class TestPrintUtilWithVerySmallFloatLessThanOneAndBigN(
    TestCase, PrintUtilBaseTestCase
):
    a = 3.289e-5
    n = 4
    relevant_precision = 5
    precise_string = "3.2890e-05"

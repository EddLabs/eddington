from eddington_test import MetaTestCase

from eddington_core.print_util import to_precise_string, to_relevant_precision


class PrintUtilMetaTestCase(MetaTestCase):
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


class TestPrintUtilWithPositiveInteger(metaclass=PrintUtilMetaTestCase):
    a = 14
    n = 2
    relevant_precision = 0
    precise_string = "14.00"


class TestPrintUtilWithSmallInteger(metaclass=PrintUtilMetaTestCase):
    a = 3
    n = 4
    relevant_precision = 0
    precise_string = "3.0000"


class TestPrintUtilWithNegativeInteger(metaclass=PrintUtilMetaTestCase):
    a = -14
    n = 2
    relevant_precision = 0
    precise_string = "-14.00"


class TestPrintUtilWithOne(metaclass=PrintUtilMetaTestCase):
    a = 1
    n = 3
    relevant_precision = 0
    precise_string = "1.000"


class TestPrintUtilWithNegativeOne(metaclass=PrintUtilMetaTestCase):
    a = -1
    n = 3
    relevant_precision = 0
    precise_string = "-1.000"


class TestPrintUtilWithZero(metaclass=PrintUtilMetaTestCase):
    a = 0
    n = 3
    relevant_precision = 0
    precise_string = "0.000"


class TestPrintUtilWithFloatBiggerThanOneAndSmallN(metaclass=PrintUtilMetaTestCase):
    a = 3.141592
    n = 2
    relevant_precision = 0
    precise_string = "3.14"


class TestPrintUtilWithFloatBiggerThanOneAndBigN(metaclass=PrintUtilMetaTestCase):
    a = 3.52
    n = 5
    relevant_precision = 0
    precise_string = "3.52000"


class TestPrintUtilWithFloatLessThanOneAndSmallN(metaclass=PrintUtilMetaTestCase):
    a = 0.3289
    n = 1
    relevant_precision = 1
    precise_string = "0.33"


class TestPrintUtilWithFloatLessThanOneAndBigN(metaclass=PrintUtilMetaTestCase):
    a = 0.52
    n = 3
    relevant_precision = 1
    precise_string = "0.5200"


class TestPrintUtilWithVerySmallFloatLessThanOneAndSmallN(
    metaclass=PrintUtilMetaTestCase
):
    a = 3.289e-5
    n = 1
    relevant_precision = 5
    precise_string = "3.3e-05"


class TestPrintUtilWithVerySmallFloatLessThanOneAndBigN(
    metaclass=PrintUtilMetaTestCase
):
    a = 3.289e-5
    n = 4
    relevant_precision = 5
    precise_string = "3.2890e-05"

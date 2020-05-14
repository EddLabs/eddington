import math
from unittest import TestCase

from eddington_core import FitFunction, FitFunctionLoadError, FitFunctionsRegistry
from mock import patch


class FitFunctionFromStringBaseTestCase(type):
    decimal = 5
    uuid = "1234"
    name = None
    expected_name = "dummy-1234"

    def __new__(mcs, name, bases, dct):
        dct.update(
            dict(
                decimal=mcs.decimal,
                uuid=mcs.uuid,
                name=mcs.name,
                expected_name=mcs.expected_name,
                setUp=mcs.setUp,
                tearDown=mcs.tearDown,
                test_name=mcs.test_name,
                test_title_name=mcs.test_title_name,
                test_parameters_number=mcs.test_parameters_number,
                test_syntax=mcs.test_syntax,
                test_value=mcs.test_value,
                test_save=mcs.test_save,
                test_is_costumed=mcs.test_is_costumed,
            )
        )
        return type(name, (TestCase, *bases), dct)

    def setUp(self):
        uuid4_patcher = patch("uuid.uuid4")
        uuid4 = uuid4_patcher.start()
        uuid4.return_value = self.uuid
        self.addCleanup(uuid4_patcher.stop)
        self.func = FitFunction.from_string(self.syntax, name=self.name, save=self.save)

    def tearDown(self):
        if FitFunctionsRegistry.exists(self.func.name):
            FitFunctionsRegistry.remove(self.func.name)

    def test_name(self):
        self.assertEqual(
            self.expected_name,
            self.func.name,
            msg="Function name is different than expected",
        )

    def test_title_name(self):
        self.assertEqual(
            "Costumed Function",
            self.func.title_name,
            msg="Function title name is different than expected",
        )

    def test_parameters_number(self):
        self.assertEqual(
            self.n,
            self.func.n,
            msg="Function parameters number is different than expected",
        )

    def test_syntax(self):
        self.assertEqual(
            self.syntax,
            self.func.syntax,
            msg="Function syntax is different than expected",
        )

    def test_value(self):
        self.assertAlmostEqual(
            self.expected_value,
            self.func(self.a, self.x),
            places=self.decimal,
            msg="Function value is different than expected",
        )

    def test_save(self):
        if self.save:
            self.assertTrue(
                FitFunctionsRegistry.exists(self.func.name),
                msg="Function should exist in registry",
            )
        else:
            self.assertFalse(
                FitFunctionsRegistry.exists(self.func.name),
                msg="Function should not exist in registry",
            )

    def test_is_costumed(self):
        self.assertTrue(
            self.func.is_costumed(), msg="Functions from strings must be costumed"
        )


class TestLoadFunctionFromStringWithoutName(
    metaclass=FitFunctionFromStringBaseTestCase
):
    syntax = "a[0] + a[2] * x + sin(a[1] * x)"
    save = False
    n = 3
    a = [1, math.pi, 2]
    x = 2
    expected_value = 5


class TestLoadFunctionFromStringWithIntegralFunction(
    metaclass=FitFunctionFromStringBaseTestCase
):
    syntax = "a[0] * gamma(a[1] * x) + a[2]"
    save = False
    n = 3
    a = [2, 1, 4]
    x = 6
    expected_value = 244


class TestLoadFunctionFromStringWithName(metaclass=FitFunctionFromStringBaseTestCase):
    name = "a_very_cool_function"
    expected_name = name
    syntax = "a[0] * exp(a[1] * x)"
    save = False
    n = 2
    a = [5, 1]
    x = 1
    expected_value = 5 * math.e


class TestLoadFunctionFromStringWithSave(metaclass=FitFunctionFromStringBaseTestCase):
    syntax = "a[0] * cos(a[1] * x + a[2]) + a[3]"
    save = True
    n = 4
    a = [2, 0.5 * math.pi, math.pi, 2]
    x = 3
    expected_value = 2


class TestLoadFunctionFromStringWithSyntaxError(TestCase):
    def test_syntax_error(self):
        syntax = "a[0] + x *"
        self.assertRaises(
            FitFunctionLoadError, FitFunction.from_string, syntax,
        )

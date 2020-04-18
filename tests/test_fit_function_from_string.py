import math
from unittest import TestCase
from mock import patch

from eddington_core import FitFunctionLoadError, FitFunctionsRegistry, FitFunction


class FitFunctionFromStringBaseTestCase:
    decimal = 5
    uuid = "1234"
    name = None
    expected_name = "dummy-1234"

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
    TestCase, FitFunctionFromStringBaseTestCase
):
    syntax = "a[0] + a[2] * x + sin(a[1] * x)"
    save = False
    n = 3
    a = [1, math.pi, 2]
    x = 2
    expected_value = 5

    def setUp(self):
        FitFunctionFromStringBaseTestCase.setUp(self)

    def tearDown(self):
        FitFunctionFromStringBaseTestCase.tearDown(self)


class TestLoadFunctionFromStringWithIntegralFunction(
    TestCase, FitFunctionFromStringBaseTestCase
):
    syntax = "a[0] * gamma(a[1] * x) + a[2]"
    save = False
    n = 3
    a = [2, 1, 4]
    x = 6
    expected_value = 244

    def setUp(self):
        FitFunctionFromStringBaseTestCase.setUp(self)

    def tearDown(self):
        FitFunctionFromStringBaseTestCase.tearDown(self)


class TestLoadFunctionFromStringWithName(TestCase, FitFunctionFromStringBaseTestCase):
    name = "a_very_cool_function"
    expected_name = name
    syntax = "a[0] * exp(a[1] * x)"
    save = False
    n = 2
    a = [5, 1]
    x = 1
    expected_value = 5 * math.e

    def setUp(self):
        FitFunctionFromStringBaseTestCase.setUp(self)

    def tearDown(self):
        FitFunctionFromStringBaseTestCase.tearDown(self)


class TestLoadFunctionFromStringWithSave(TestCase, FitFunctionFromStringBaseTestCase):
    syntax = "a[0] * cos(a[1] * x + a[2]) + a[3]"
    save = True
    n = 4
    a = [2, 0.5 * math.pi, math.pi, 2]
    x = 3
    expected_value = 2

    def setUp(self):
        FitFunctionFromStringBaseTestCase.setUp(self)

    def tearDown(self):
        FitFunctionFromStringBaseTestCase.tearDown(self)


class TestLoadFunctionFromStringWithSyntaxError(TestCase):
    def test_syntax_error(self):
        syntax = "a[0] + x *"
        self.assertRaises(
            FitFunctionLoadError, FitFunction.from_string, syntax,
        )

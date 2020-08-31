from eddington import FitResult, fit_result_pretty_string

FIT_RESULT = FitResult(
    a=[1.1, -2.98e-4, 9.1],
    aerr=[0.1, 7.65e-5, 0],
    acov=[
        [0.01, 2.3, 6.2],
        [2.3, 0.988, 1.1],
        [6.2, 1.1, 3],
    ],
    chi2=8.276,
    degrees_of_freedom=5,
)


def test_simple_pretty_string():
    expected_pretty_string = """Results:
========

Initial parameters' values:
\t1.0 3.0 5.2
Fitted parameters' values:
\ta[0] = 1.100 \u00B1 0.1000 (9.091% error)
\ta[1] = -2.980e-04 \u00B1 7.650e-05 (25.671% error)
\ta[2] = 9.100 \u00B1 0.000 (0.000% error)
Fitted parameters covariance:
[[0.01  2.3   6.2  ]
 [2.3   0.988 1.1  ]
 [6.2   1.1   3.   ]]
Chi squared: 8.276
Degrees of freedom: 5
Chi squared reduced: 1.655
P-probability: 0.1417
"""
    pretty_string = fit_result_pretty_string(FIT_RESULT, a0=[1.0, 3.0, 5.2])
    assert (
        pretty_string == expected_pretty_string
    ), "Pretty string is different than expected."


def test_pretty_string_without_initial_guess():
    expected_pretty_string = """Results:
========

Fitted parameters' values:
\ta[0] = 1.100 \u00B1 0.1000 (9.091% error)
\ta[1] = -2.980e-04 \u00B1 7.650e-05 (25.671% error)
\ta[2] = 9.100 \u00B1 0.000 (0.000% error)
Fitted parameters covariance:
[[0.01  2.3   6.2  ]
 [2.3   0.988 1.1  ]
 [6.2   1.1   3.   ]]
Chi squared: 8.276
Degrees of freedom: 5
Chi squared reduced: 1.655
P-probability: 0.1417
"""
    pretty_string = fit_result_pretty_string(FIT_RESULT)
    assert (
        pretty_string == expected_pretty_string
    ), "Pretty string is different than expected."

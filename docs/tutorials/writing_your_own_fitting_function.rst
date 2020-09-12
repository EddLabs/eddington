.. writing_your_own_fitting_function:

.. role:: python(code)
   :language: python

Writing Your Own Fitting Function
=================================

When Should You Do It?
----------------------

Even though Eddington offers many default fitting functions, sometimes you may want to
customize your own fitting function.

Consider the following case: You conduct an experiment to demonstrate the *Thin Lens
Equation*:

.. math::

    \frac{1}{u}+\frac{1}{v} = \frac{1}{f}

After rearranging the equation you get:

.. math::

    v = \frac{uf}{u - f}

You have data records of :math:`v` and :math:`u` and you want to estimate :math:`f`.
You **can** use our *hyperbolic* fitting function, but in order to find
:math:`f` You'll have to do some more calculations with respect to the errors of
the parameters you've found.

A somewhat easier approach would be to use the following fitting function:

.. math::

    y=\frac{a_0x}{x-a_0}+a_1

Now, after you fit the data, you get :math:`f` directly (which equals to :math:`a_0`)

Since this fitting function is not implemented by default, you'd have to implement
it yourself.

Basic implementation
--------------------

A basic implementation of the fitting function presented in the example above would
look like this:

.. code:: python

    from eddington import fit_function

    @fit_function(n=2)
    def lens(a, x):
        return (a[0] * x) / (x - a[0]) + a[1]

We wrap the :python:`lens` fitting function with the :python:`fit_function` decorator
in order to indicate that this function is actually a fitting function. the :python:`n`
variable indicates how many parameters the fitting function expects. In our example,
we expect 2 parameters: :python:`a[0]` which is :math:`f`, and :python:`a[1]` which
encapsulates the systematic errors in our :math:`v` samples.

.. note::

    The inputs of the fitting function are :python:`a` which is the parameters vector
    and :python:`x` which is the free variable. While :python:`a` can be from any
    array-like type such as :python:`list`, :python:`tuple`, :python:`numpy.ndarray`,
    etc. :python:`x` can be both a :python:`numpy.ndarray` and :python:`float`.

Now, we can use the fitting function we've created in order to fit the data:

.. code:: python

    from eddington import FittingData, fit_to_data

    fitting_data = FittingData.read_from_csv("/path/to/data.csv")  # Load data from file.
    fitting_result = fit_to_data(fitting_data, lens)  # Do the actual fitting
    print(fitting_result)  # Print the results

This usage is more than enough for most use-cases.

Derivatives
-----------

Sometimes, you wish to get an accurate fit, and **fast**. One way to achieve that is
to add derivatives to the fitting function. In our example, we have the following
derivatives:

:math:`x` *derivative* -

.. math::

    \frac{\partial y}{\partial x}=-\frac{a_0^2}{(x-a_0)^2}

:math:`a_0` *derivative* -

.. math::

    \frac{\partial y}{\partial a_0}=\frac{x^2}{(x-a_0)^2}

:math:`a_1` *derivative* -

.. math::

    \frac{\partial y}{\partial a_1}=1

In order to add those derivatives to the fitting function, we should add the
:python:`x_derivative` and :python:`a_derivative` to the :python:`fit_function`
decorator. In our example:

.. code:: python

    import numpy as np
    from eddington import fit_function, FittingData, fit_to_data


    @fit_function(
        n=2,
        x_derivative=lambda a, x: -np.power(a[0], 2) / np.power(x - a[0], 2),
        a_derivative=lambda a, x: np.stack(
            [
                np.power(x, 2) / np.power(x - a[0], 2),
                np.ones(shape=np.shape(x)),
            ]
        ),
    )
    def lens(a, x):
        return (a[0] * x) / (x - a[0]) + a[1]

.. note::

    When implementing the derivatives pay attention that you take :python:`a` as the
    first parameter and :python:`x` as the second. Moreover, make sure that the
    *dimension* of the output :python:`x_derivative` returns a :python:`numpy.ndarray`
    with dimension similar to :python:`x`, while :python:`a_derivative` returns a
    :python:`numpy.ndarray` with dimension equal to :python:`x` dimension times
    :python:`a` dimension.


The Fitting Functions Registry
-------------------------------

By default, creating a new fitting function adds it automatically to the
`FitFunctionsRegistry`, a singleton containing all fitting functions.
Once the fitting function you've created is imported (for example, in the *__init__.py*
file) it can be loaded from the registry in the following way:

.. code:: python

    from eddington import FitFunctionsRegistry

    fit_func = FitFunctionsRegistry.load("lens")

If you wish to specify a different name to the fitting function by which it can be
loaded from the registry, use the `name` parameter in the `fit_function` decorator
in the following way:

.. code:: python

    from eddington import fit_function, FitFunctionsRegistry

    @fit_function(n=2, name="my_amazing_func")
    def lens(a, x):
        return (a[0] * x) / (x - a[0]) + a[1]

    fit_func = FitFunctionsRegistry.load("my_amazing_func")  # Returns the "lens" function

If you expect others to use your new fitting function, consider adding a `syntax` string
indicating how the fitting functions fit the data. This can be printed out when needed.
For example:

.. code:: python

    from eddington import fit_function, FitFunctionsRegistry

    @fit_function(n=2, syntax="(a[0] * x) / (x - a[0]) + a[1]")
    def lens(a, x):
        return (a[0] * x) / (x - a[0]) + a[1]

    ...

    fit_func = FitFunctionsRegistry.load("lens")
    print(f"Syntax is: {fit_func.syntax}")  # Prints out the defined syntax

Lastly, if you wish the fitting function to not be saved into the registry, specify
:python:`save=False` in the `fit_function` decorator. For example:


.. code:: python

    from eddington import fit_function

    @fit_function(n=2, save=False)
    def lens(a, x):
        return (a[0] * x) / (x - a[0]) + a[1]

As mentioned earlier, by default `save` is set to :python:`True`.

.. warning::

    Two functions cannot be saved into the registry under the same name. Make sure that
    every new fitting function you write has a unique name, which is not one of the
    default fitting functions or another custom fitting function you expect to use
    in your code.
.. contribution_guide:

Contribution Guide
==================

Thank you for choosing to spend your time contributing to the Eddington platform.
Contributors are the living, beating heart of any open-source project, and we really
appreciate each and every contribution made to our code-base, big or small.

Here is a relatively short guide on how to contribute code to Eddington. Please follow
the next steps carefully when making a pull request.

Step 1 – Writing Your Code
--------------------------

Like any other open-source project in Github, contribution to Eddington is done via a
`pull-request`_ (PR). Fork the desired repository, open a feature branch, and write
your code in it.

When writing code, please pay attention that you:

1. Make sure your *master* branch is `up-to-date`_ with the latest changes in the Eddington platform, and make your feature branch based upon it. This will help you avoid merge conflicts.
2. Write your code clearly, with self-explainable variables, functions and classes.
3. Reuse existing code when possible.
4. Document new functions, classes, and modules (especially if they’re public).

The code reviews you’ll receive would often address the following guidelines, as well
as any existing design issues.

Step 2 – Testing Your Code
--------------------------

In the Eddington platform, we believe in 100% test coverage, and we enforce it
throughout our repositories! If you add new functionalities or change an existing
functionality, you must test the new ability with a unit test
(or better yet – **unit tests**).

We use pytest_ as our testing platform. Some of our tests use the `pytest-case`_
and `pytest-mock`_ libraries. Feel free to use those libraries as well as other testing
libraries whenever needed.

In order to run the unit tests and see the code coverage, you should use tox_:

1. Install *tox* with :code:`pip install tox`
2. Go to the main repository directory and run :code:`tox -e py`
3. This will run the unit tests and show the coverage report. Make sure the code passes and all lines are covered

We will never compromise on code coverage and/or extensive unit testing.

Step 3 – Cleaning Your Code
---------------------------

Writing a working code can sometimes be really hard, but writing a **clean** code is always
harder.

Here on the Eddington platform we believe that code should be clean, and we want to
make sure that writing clean code is as easy as possible. We do that by using automatic
tools that would help you achieve that along the way.

We use state-of-the-art static code analysis tools such as *black*, *flake8*, *pylint*,
*mypy*, *pydocstyle*, etc. Statue_ is orchestrating all these tools by running each of
them on all of our code-base.

In order to use *Statue*, follow the next steps:

1. Run :code:`pip install statue`.
2. Run :code:`statue install`. If needed, this command will install missing packages.
3. Go to the main repository directory and run :code:`statue run --context format`. This will change your code to fit styling guidelines. Save the changes in a commit or append them to an existing commit.
4. Run :code:`statue run` again (now without any arguments) and it will check if there are any issues that it wasn't able to solve on its own. If there are any errors, fix them.
5. Save all changes in a commit or append them to an existing commit.

You may find some of the errors presented by those tools tedious or irrelevant,
but rest assured that we take those errors seriously.

If you think that in a specific line an error should be ignored (using :code:`# noqa`
or :code:`# pylint: disable` for example), please make sure that this skip is justified
before applying it.

Step 4 – Re-running everything
------------------------------

Go to the main repository directory and run :code:`tox` (without any arguments).
This will re-run all the different checks we use in Eddington.

Make sure that everything is checked out before continuing to the next step.

Step 5 – Adding Yourself to the Acknowledgment File
----------------------------------------------------

We acknowledge each and every one of our contributors in *docs/acknowledgment.rst*.
Add your name to the contributors file, keeping alphabetical order.


Step 6 – Receiving a Code Review and Merging
---------------------------------------------

Push the branch and open a PR. We will make our best efforts to review your PR as soon
as possible.

Once you receive a code-review, address the issues presented to you by changing the
code or commenting back. Once all the issues are resolved, your PR will be merged to
master!

.. _pull-request: https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-requests
.. _up-to-date: https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/syncing-a-fork
.. _pytest: https://docs.pytest.org/en/stable/
.. _pytest-case: https://smarie.github.io/python-pytest-cases/
.. _pytest-mock: https://github.com/pytest-dev/pytest-mock/
.. _tox: https://tox.readthedocs.io/en/latest/
.. _statue: https://github.com/saroad2/statue

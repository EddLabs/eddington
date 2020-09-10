.. _writing_data_file:

Writing Data Files
===================


Data File Formats
-----------------

Eddington excepts data files in 3 formats: CSV, Excel, and Json. In this tutorial we
will walk through the syntax of each format.

CSV File
~~~~~~~~

CSV data files should have a headers row followed by records rows, each column represent
a specific value of the record. For example:

::

    a,delta a,b,delta b,c,delta c
    10,0.5,16,1,100,14
    20,1,29,1.3,401,10
    30,1.2,47,0.8,910,11
    40,0.3,56,2,1559,8
    50,0.4,70,1.1,2480,10
    60,1.1,92,0.2,3623,5
    70,1.3,100,2,4910,16

Excel File
~~~~~~~~~~~

Those instructions are the same for Excel files:

.. figure:: /_static/excel_data.png

Json File
~~~~~~~~~

As for Json files, it should be written as a dictionary, with each header as the key
mapped to it's values. For example:

.. code:: json

    {
      "a": [10, 20, 30, 40, 50, 60, 70],
      "delta_a": [0.5, 1.0, 1.2, 0.3, 0.4, 1.1, 1.3],
      "b": [16.0, 29.0, 47.0, 56.0, 70.0, 92.0, 100.0],
      "delta_b": [1.0, 1.3, 0.8, 2.0, 1.1, 0.2, 2.0],
      "c": [100.0, 401.0, 910.0, 1559.0, 2480.0, 3623.0, 4910.0],
      "delta_d": [14.0, 10.0, 11.0, 8.0, 10.0, 5.0, 16.0]
    }

.. warning::

    The order of each values list should be the orders of the records. Mismatching
    the order may cause fitting errors.

Specify The Data Columns
------------------------

Whatever the data file format you choose, you should specify the columns to be used for
the fitting. By default the first 4 columns are used as the *x*, *x error*, *y* and
*y error* in that order. That means that in our example "a" will be taken as *x*,
"delta_a" will be taken as the *x error*, "b" will be taken as *y* and "delta_b" will
be taken as the *y error*.

Common Errors
--------------

If you encounter a loading error while trying to load a data file, it may be caused by
one of the following problems:

* One or more of your records has non-float value or it is blank.
* In Excel files, you may have added an additional cell outside of the records table.
* In Excel and CSV files, you might have missed adding a headers row
* In Json, one or more of your columns has different length than the others (which means a record is missing a value).

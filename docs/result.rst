******
Result
******

This module contains the `data <https://docs.python.org/3/library/dataclasses.html>`_ class which acts as a container for all the results and parameters calculated or needed for the analysis code.
Default value for numerical values is -99. If in the final results -99 appears this means that the value was either not clacluated or there has been an error with that part of the analysis.
Default value for strings is an empty string.

.. autoclass:: pawlikMorphLSST.result.Result
    :members:
    :undoc-members:
    :noindex:
Arbor
=====

|testbadge| |zlatest|

.. |testbadge| image:: https://github.com/arbor-sim/arbor/actions/workflows/basic.yml/badge.svg
    :target: https://github.com/arbor-sim/arbor/actions/workflows/basic.yml

Welcome to the documentation for Arbor, the multi-compartment neural network simulation library.

Documentation organisation
--------------------------

* :ref:`internals-overview` describes Arbor code that is not user-facing; convenience classes, architecture abstractions, etc.
* Contributions to Arbor are very welcome! Under :ref:`contribindex` describe conventions and procedures for all kinds of contributions.

Citing Arbor
------------

The Arbor software can be cited by version via Zenodo or via Arbors introductory paper.

Latest version
    |zlatest|

Version 0.5.2
    |z052|

    .. code-block:: latex

        @software{nora_abi_akar_2021_4428108,
        author       = {Nora {Abi Akar} and
                        John Biddiscombe and
                        Benjamin Cumming and
                        Felix Huber and
                        Marko Kabic and
                        Vasileios Karakasis and
                        Wouter Klijn and
                        Anne Küsters and
                        Alexander Peyser and
                        Stuart Yates and
                        Thorsten Hater and
                        Brent Huisman and
                        Sebastian Schmitt},
        title        = {arbor-sim/arbor: Arbor Library v0.5},
        month        = jan,
        year         = 2021,
        publisher    = {Zenodo},
        version      = {v0.5},
        doi          = {10.5281/zenodo.4428108},
        url          = {https://doi.org/10.5281/zenodo.4428108}
        }

Version 0.2
    |z02|

Version 0.1
    |z01|

Acknowledgements
----------------

This research has received funding from the European Unions Horizon 2020 Framework Programme for Research and
Innovation under the Specific Grant Agreement No. 720270 (Human Brain Project SGA1), Specific Grant Agreement
No. 785907 (Human Brain Project SGA2), and Specific Grant Agreement No. 945539 (Human Brain Project SGA3).

Arbor is an `eBrains project <https://ebrains.eu/service/arbor/>`_.

A full list of our software attributions can be found `here <https://github.com/arbor-sim/arbor/blob/master/ATTRIBUTIONS.md>`_.

.. toctree::
   :caption: API reference:
   :maxdepth: 1

   internals/index

.. toctree::
   :caption: Contributing:
   :maxdepth: 1

   contrib/index
   contrib/pr
   contrib/coding-style
   contrib/doc
   contrib/test

.. meta::
   :google-site-verification: KbkW8d9MLsBFZz8Ry0tfcQRkHsgxzkECCahcyRSjWDo

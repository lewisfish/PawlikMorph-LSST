************
LSST Support
************

In order to be able ingest images via the LSST pipeline you must first install the LSST pipeline.

**Note LSST support is currently experimental.**
The LSST API is currently not finalised so using the latest version of the LSST pipeline may break the code. 
If in doubt try and use version 19, the last version this code was tested against.

To install the LSST pipeline please refer to `<https://pipelines.lsst.io/#installation>`_. We recommend using `Docker <https://pipelines.lsst.io/install/docker.html>`_ as the installation method for the LSST pipeline.
The following instructions assume that Docker has been used to install the LSST pipeline.

To use run pawlikMorph-LSST with the LSST pipeline first mount the directory where the pawlikMorph-LSST code is stored (i.e the directory above pawlikMorphLSST/).

.. code-block:: console

    $ docker run -it -v `pwd`:/home/lsst/mnt lsstsqre/centos:7-stack-lsst_distrib-v19_0_0

Setup the LSST environment using which ever shell you are using (if in doubt try .bash)

.. code-block:: console

    $ source loadLSST.bash
    $ setup lsst_distrib


Next you need to change directory into the directory with the code and data

.. code-block:: console
    
    $ cd
    $ cd mnt/

Finally you need to install the dependacies that do not come with the LSST Docker image

.. code-block:: console
    
    $ pip install numba parsl scikit-image photutils

If everything was correctly installed then running

.. code-block:: console

    $ python imganalysis --help

Should output the help for the imganalysis script.


For more help on Docker `<https://docs.docker.com/get-started/>`_.
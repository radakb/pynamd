PyNAMD
======

Python tools for NAMD

Description
===========

PyNAMD is a package meant to provide a flexible, lightweight interface to NAnoscale Molecular Dynamics ([NAMD](http://www.ks.uiuc.edu/Research/namd/)) input and output. Since many powerful and mature packages exist for trajectory analysis, the focus is almost exclusively on energy based output.

The PyNAMD library is also accompanied by several scripts for common tasks useful in molecular dynamics (MD) simulations, such as rapidly computing averages and fluctuations - all directly from NAMD output. Developments are ongoing to provide considerably more complicated analysis tools such as multistate reweighting methods (e.g., WHAM) for generalized ensemble, replica exchange, and stratified umbrella sampling simulations.

Installation
============

To install PyNAMD, clone this git repository and change into the new "pynamd" directory. PyNAMD can then be built with the command

```
python setup.py install
```

If you are using the Python distribution that comes with your operating system, you may need to run the above command with administrative privileges or else add the ``--user`` flag -- the latter is recommended. It is expected that this project will utilize a considerable amount of "bleeding edge" numerical tools, so it would also be a good idea to use a customizable Python environment that has the latest numpy and scipy, such as that provided by [Anaconda](https://www.continuum.io/downloads).


Examples
========

Check back soon.

Authors and Contributors
========================

* Brian Radak | brian.radak@gmail.com | brian.radak@anl.gov


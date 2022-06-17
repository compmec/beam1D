# Structures

[![PyPi Version](https://img.shields.io/pypi/v/compmec-strct.svg?style=flat-square)](https://pypi.org/project/compmec-strct/)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/compmec-strct.svg?style=flat-square)](https://pypi.org/project/compmec-strct/)
![Tests](https://github.com/compmec/strct/actions/workflows/tests.yml/badge.svg)

Solves struture problems using FEM. 

* 1D Elements
    * Cable (Only traction)
    * Truss (Traction/Compression)
    * Beam
        * Euler-Bernoulli
        * Timoshenko
* 2D Elements
    * Plate

## How to use it

There are many **Python Notebooks** in the folder  ```examples```.

For simple examples, see
* 1D Elements
    * Cable
    * Truss
    * Beam
        * Euler-Bernoulli
        * Timoshenko
* 2D Elements
    * Plate

## Install

This library is available in [PyPI][pypilink]. To install it

```
pip install compmec-strct
```

Or install it manually

```
git clone https://github.com/compmec/strct
cd strct
pip install -e .
```

To verify if everything works in your machine, type the command in the main folder

```
pytest
```

## Documentation

In progress

## Contribute

Please use the [Issues][issueslink] or refer to the email ```compmecgit@gmail.com```


[pypilink]: https://pypi.org/project/compmec-strct/
[issueslink]: https://github.com/compmec/strct/issues
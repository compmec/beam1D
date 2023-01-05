[![PyPI Version][pypi-image]][pypi-url]
[![Build Status][build-image]][build-url]
[![Code Coverage][coverage-image]][coverage-url]
[![][versions-image]][versions-url]

# Structures

Compute fields (displacement/force/momentum) of structures using Finite Element Method. 

* 1D Elements
    * Truss (Traction/Compression)
    * Beam
        * Euler-Bernoulli
        * Timoshenko (Needs implementation)

## How to use it

There are many **Python Notebooks** in the folder  ```examples```.


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

## Documentation

In progress. See the examples.

## Contribute

Please use the [Issues][issueslink] or refer to the email ```compmecgit@gmail.com```


[pypi-image]: https://img.shields.io/pypi/v/compmec-strct
[pypi-url]: https://pypi.org/project/compmec-strct/
[build-image]: https://github.com/compmec/strct/actions/workflows/build.yaml/badge.svg
[build-url]: https://github.com/compmec/strct/actions/workflows/build.yaml
[coverage-image]: https://codecov.io/gh/compmec/strct/branch/main/graph/badge.svg
[coverage-url]: https://codecov.io/gh/compmec/strct/
[versions-image]: https://img.shields.io/pypi/pyversions/compmec-strct.svg?style=flat-square
[versions-url]: https://pypi.org/project/compmec-strct/
[pypilink]: https://pypi.org/project/compmec-strct/
[issueslink]: https://github.com/compmec/strct/issues

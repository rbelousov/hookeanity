# Hookeanity

Illustration of the overdamped fluctuating elastodynamics model.

## Installation

Get the source code and install using pip, e.g.

```bash
git clone https://git.embl.de/rbelouso/hookeanity.git hookeanity && cd hookeanity
pip install .
```

To try out examples available in the `examples` directory, replace the last line with

```bash
pip install .[examples]
```

For development install in the editable mode with `dev` dependencies

```bash
pip install -e .[dev]
```

A complete set of optional dependencies: `tests` (unit tests only), `dev` (tests and development dependencies), `doc` (documentation generators), `examples` (examples).


## Unit tests

Unit tests are implemented with `pytest` package. If the optional dependencies for tests are installed (`tests` or `dev`), to run the test invoke `pytest` in the command line.


## Wolfram paclet `SphericalMatern`

For illustration purposes this package also provides a paclet for Wolfram Mathematica that implements Sobolev-Matérn interpolation with unweighted cardinal functions on a sphere.

To install the paclet `SphericalMatern` run

```bash
wolframscript -file `SphericalMatern/install.wls`
```

To uninstall the paclet use

```bash
wolframscript -file `SphericalMatern/uninstall.wls`
```

Tests can be be executed with 

```bash
wolframscript -file `SphericalMatern/runTests.wls`
```

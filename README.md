# Hookean elastodynamics

## Installation

Get the source code and install using pip, e.g.

```bash
git clone https://git.embl.de/rbelouso/hookeanity.git hookeanity && cd hookeanity
pip install .
```

For development you may replace the last line with

```bash
pip install -e .[dev]
```

To include documentation and examples, use
```bash
pip install -e .[doc,examples]
```

## Wolfram paclet `SphericalMatern`

To install the paclet `SphericalMatern` run
```bash
wolframscript -file `SphericalMatern/install.wls`
```

To uninstall the paclet use
```
wolframscript -file `SphericalMatern/uninstall.wls`
```

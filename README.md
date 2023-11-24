# quant-research
Quantitative finance research python libraries

# To create a package

1. Update the pyproject.toml's version number

1. Install dependencies

```sh
python3 -m pip install --upgrade build
```

1. Create a build

```sh
python3 -m build
```

1. Upload archives

```sh
python3 -m pip install --upgrade twine
python3 -m twine upload --repository testpypi dist/*
```
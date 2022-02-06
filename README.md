# How Non-Experts Perceive and Teach with Machine Learning Uncertainties

This repository is the official implementation of [How Non-Experts Perceive and Teach with Machine Learning Uncertainties]().
It contains:
- `benchmark`: [jupyter notebooks](https://jupyter.org/) presenting the benchmark presented in section 3.
- `app`: the [Marcelle](https://www.marcelle.dev) source code of the application used in the user study presented in section 4.
- `analysis`: the quantitative analysis of the user study logs. Results are presented in section 5.
- `setup`: contains python configurations of the project
- `common`: contains common functions used both in the app, the benchmark and the analysis.

## Requirements

### Python requirements
To install requirements in a conda environment:
On MacOS:

```setup
pip install -r setup/conda_env_macos.yml
```

On Linux:
```setup
pip install -r setup/conda_env_linux.yml
```

### Application requirements

To install the app:
```setup
cd app
npm install
```
or 
```setup
cd app
yarn install
```

See the README in `app` for more details.

To install Marcelle, please refers to [the official website](https://www.marcelle.dev)

## Contributing

The code here might be incomplete and not systematically documented. It aims at making the research process more transparent for reviewers and peers. Please contact me if you have any suggestions or find any problems.


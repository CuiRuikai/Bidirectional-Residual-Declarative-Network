# Deep Declarative Networks

This is the ReadMe file in the DDN repo. We use this library to build our Classification Model.

To see more about it, please visit [here](https://github.com/anucvml/ddn)

Deep Declarative Networks (DDNs) are a class of deep learning model that allows for optimization problems
to be embedded within an end-to-end learnable network. This repository maintains code,
[tutorials](https://nbviewer.jupyter.org/github/anucvml/ddn/tree/master/tutorials/) and other
[resources](https://github.com/anucvml/ddn/wiki/Resources) for developing and understanding DDN models.

You can find more details in [this paper](https://arxiv.org/abs/1909.04866), which if you would like to
reference in your research please cite as:
```
@techreport{Gould:PrePrint2019,
  author      = {Stephen Gould and
                 Richard Hartley and
                 Dylan Campbell},
  title       = {Deep Declarative Networks: A New Hope},
  eprint      = {arXiv:1909.04866},
  institution = {Australian National University (arXiv:1909.04866)},
  month       = {Sep},
  year        = {2019}
}
```

Reference (PyTorch) applications for image and point cloud classification can be found under the `apps`
directory. See the `README` files therein for instructions on installation and how to run.

## License

The `ddn` library is distributed under the MIT license. See the [LICENSE](LICENSE) file for details.


# `ddn` Package

This document provides a brief description of the modules and utilities within the `ddn` package.
For an overview of deep declarative network concepts and demonstration of using the library see the
[tutorials](https://nbviewer.jupyter.org/github/anucvml/ddn/tree/master/tutorials/).

## Basic

The `ddn.basic` package contains standard python code for experimenting with deep declarative nodes. The
implementation assumes that all inputs and outputs are vectors (or more complicated data structures
have been vectorized).

* `ddn.basic.composition`: implements wrapper code for composing nodes in series or parallel (i.e., building a network).
* `ddn.basic.node`: defines the interface for data processing nodes and declarative nodes.
* `ddn.basic.robust_nodes`: implements nodes for robust pooling.
* `ddn.basic.sampls_nodes`: provided examples of deep declarative nodes used for testing and in the tutorials.


## PyTorch

The `ddn.pytorch` package includes efficient implementations of deeep declarative nodes suitable for including
in an end-to-end learnable model. The code builds on the PyTorch framework and conventions.

* `ddn.geometry_utilities`: utility functions for geometry applications.
* `ddn.pytorch.node`: defines the PyTorch interface for data processing nodes and declarative nodes.
* `ddn.pytorch.pnp_node`: differentiable projection-n-point algorithm.
* `ddn.pytorch.projections`: differentiable Euclidean projection layers onto Lp balls and spheres.
* `ddn.pytorch.robostpool`: differentiable robust pooling layers.
* `ddn.pytorch.sample_nodes`: simple example implementations of deep declarative nodes for PyTorch.

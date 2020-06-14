# DEEP DECLARATIVE NODES
# Defines the interface for data processing nodes and declarative nodes. The implementation here is kept simple
# with inputs and outputs assumed to be vectors. There is no distinction between data and parameters and no
# concept of batches. For using deep declarative nodes in a network for end-to-end learning see code in the
# `ddn.pytorch` package.
#
# Stephen Gould <stephen.gould@anu.edu.au>
# Dylan Campbell <dylan.campbell@anu.edu.au>
#

import autograd.numpy as np
import scipy as sci
from autograd import grad, jacobian
import warnings

class AbstractNode:
    """
    Minimal interface for generic data processing node that produces an output vector given an input vector.
    """

    def __init__(self, n=1, m=1):
        """
        Create a node
        :param n: dimensionality of the input (parameters)
        :param m: dimensionality of the output (optimization solution)
        """
        assert (n > 0) and (m > 0)
        self.dim_x = n # dimensionality of input variable
        self.dim_y = m # dimensionality of output variable

    def solve(self, x):
        """Computes the output of the node given the input. The second returned object provides context
        for computing the gradient if necessary. Otherwise it's None."""
        raise NotImplementedError()
        return None, None

    def gradient(self, x, y=None, ctx=None):
        """Computes the gradient of the node for given input x and, optional, output y and context cxt.
        If y or ctx is not provided then they are recomputed from x as needed."""
        raise NotImplementedError()
        return None


class AbstractDeclarativeNode(AbstractNode):
    """
    A general deep declarative node defined by an unconstrained parameterized optimization problems of the form
        minimize (over y) f(x, y)
    where x is given (as a vector) and f is a scalar-valued function. Derived classes must implement the `objective`
    and `solve` functions.
    """

    eps = 1.0e-6 # tolerance for checking that optimality conditions are satisfied

    def __init__(self, n=1, m=1):
        """
        Creates an declarative node with optimization problem implied by the objecive function. Initializes the
        partial derivatives of the objective function for use in computing gradients.
        """
        super().__init__(n, m)

        # partial derivatives of objective
        self.fY = grad(self.objective, 1)
        self.fYY = jacobian(self.fY, 1)
        self.fXY = jacobian(self.fY, 0)

    def objective(self, x, y):
        """Evaluates the objective function on a given input-output pair."""
        warnings.warn("objective function not implemented.")
        return 0.0

    def solve(self, x):
        """
        Solves the optimization problem
            y \in argmin_u f(x, u)
        and returns two outputs. The first is the optimal solution y and the second contains the context for
        computing the gradient, such as the largrange multipliers in the case of a constrained problem, or None
        if no context is available/needed.
        """

        raise NotImplementedError()
        return None, None

    def gradient(self, x, y=None, ctx=None):
        """
        Computes the gradient of the output (problem solution) with respect to the problem
        parameters. The returned gradient is an ndarray of size (self.dim_y, self.dim_x). In
        the case of 1-dimensional parameters the gradient is a vector of size (self.dim_y,).
        Can be overridden by the derived class to provide a more efficient implementation.
        """

        # compute optimal value if not already done so
        if y is None:
            y, ctx = self.solve(x)
        assert self._check_optimality_cond(x, y)

        return -1.0 * sci.linalg.solve(self.fYY(x, y), self.fXY(x, y), assume_a='pos')

    def _check_optimality_cond(self, x, y, ctx=None):
        """Checks that the problem's first-order optimality condition is satisfied."""
        return (abs(self.fY(x, y)) <= self.eps).all()


class EqConstDeclarativeNode(AbstractDeclarativeNode):
    """
    A general deep declarative node defined by a parameterized optimization problem with single (non-linear)
    equality constraint of the form
        minimize (over y) f(x, y)
        subject to        h(x, y) = 0
    where x is given (as a vector) and f and h are scalar-valued functions. Derived classes must implement the
    `objective`, `constraint` and `solve` functions.
    """

    def __init__(self, n, m):
        super().__init__(n, m)

        # partial derivatives of constraint function
        self.hY = grad(self.constraint, 1)
        self.hX = grad(self.constraint, 0)
        self.hYY = jacobian(self.hY, 1)
        self.hXY = jacobian(self.hY, 0)

    def constraint(self, x, y):
        """Evaluates the equality constraint function on a given input-output pair."""
        warnings.warn("constraint function not implemented.")
        return 0.0

    def solve(self, x):
        """
        Solves the optimization problem
            y \in argmin_u f(x, u) subject to h(x, u) = 0
        and returns the vector y. Optionally, also returns the Lagrange multiplier associated with the
        equality constraint where the Lagrangian is defined as
            L(x, y, nu) = f(x, y) - ctx['nu'] * h(x, y)
        Otherwise, should return None as second return variable.
        If the calling function only cares about the optimal solution (and not the context) then call as
            y_star, _ = self.solve(x)
        """

        raise NotImplementedError()
        return None, None

    def gradient(self, x, y=None, ctx=None):
        """Compute the gradient of the output (problem solution) with respect to the problem
        parameters. The returned gradient is an ndarray of size (prob.dim_y, prob.dim_x). In
        the case of 1-dimensional parameters the gradient is a vector of size (prob.dim_y,)."""

        # compute optimal value if not already done so
        if y is None:
            y, ctx = self.solve(x)
        assert self._check_constraints(x, y), [x, y, abs(self.constraint(x, y))]
        assert self._check_optimality_cond(x, y, ctx), [x, y, ctx]

        nu = self._get_nu_star(x, y) if (ctx is None) else ctx['nu']

        # return unconstrained gradient if nu is undefined
        if np.isnan(nu):
            return -1.0 * np.linalg.solve(self.fYY(x, y), self.fXY(x, y))

        H = self.fYY(x, y) - nu * self.hYY(x, y)
        a = self.hY(x, y)
        B = self.fXY(x, y) - nu * self.hXY(x, y)
        C = self.hX(x, y)
        try:
            v = sci.linalg.solve(H, np.concatenate((a.reshape((self.dim_y, 1)), B), axis=1), assume_a='pos')
        except:
            return np.full((self.dim_y, self.dim_x), np.nan).squeeze()
        return (np.outer(v[:, 0], (v[:, 0].dot(B) - C) / v[:, 0].dot(a)) - v[:, 1:self.dim_x + 1]).squeeze()

    def _get_nu_star(self, x, y):
        """Compute nu_star if not provided by the problem's solver."""
        indx = np.nonzero(self.hY(x, y))
        if len(indx[0]) == 0:
            return 0.0
        return self.fY(x, y)[indx[0][0]] / self.hY(x, y)[indx[0][0]]

    def _check_constraints(self, x, y):
        """Check that the problem's constraints are satisfied."""
        return abs(self.constraint(x, y)) <= self.eps

    def _check_optimality_cond(self, x, y, ctx=None):
        """Checks that the problem's first-order optimality condition is satisfied."""
        nu = self._get_nu_star(x, y) if (ctx is None) else ctx['nu']
        if np.isnan(nu):
            return (abs(self.fY(x, y)) <= self.eps).all()

        # check for invalid lagrangian (gradient of constraint zero at optimal point)
        if (abs(self.hY(x, y)) <= self.eps).all():
            warnings.warn("gradient of constraint function vanishes at the optimum.")
            return True
        return (abs(self.fY(x, y) - nu * self.hY(x, y)) <= self.eps).all()


class IneqConstDeclarativeNode(EqConstDeclarativeNode):
    """
    A general deep declarative node defined by a parameterized optimization problem with single (non-linear)
    inequality constraint of the form
        minimize (over y) f(x, y)
        subject to        h(x, y) <= 0
    where x is given (as a vector) and f and h are scalar-valued functions. Derived classes must implement the
    `objective`, `constraint` and `solve` functions.
    """

    def __init__(self, n, m):
        super().__init__(n, m)

    def _get_nu_star(self, x, y):
        """Compute nu_star if not provided by the problem's solver."""
        if np.all(np.abs(self.fY(x, y)) < self.eps):
            return np.nan # flag that unconstrained gradient should be used
        indx = np.nonzero(self.hY(x, y))
        if len(indx[0]) == 0:
            return 0.0 # still use constrained gradient
        return self.fY(x, y)[indx[0][0]] / self.hY(x, y)[indx[0][0]]

    def _check_constraints(self, x, y):
        """Check that the problem's constraints are satisfied."""
        return self.constraint(x, y) <= self.eps


class LinEqConstDeclarativeNode(AbstractDeclarativeNode):
    """
    A deep declarative node defined by a linear equality constrained parameterized optimization problem of the form:
        minimize (over y) f(x, y)
        subject to        A y = b
    where x is given. Derived classes must implement the objective and solve functions.
    """

    def __init__(self, n, m, A, b):
        super().__init__(n, m)
        assert A.shape[1] == m, "second dimension of A must match dimension of y"
        assert A.shape[0] == b.shape[0], "dimension of A must match dimension of b"
        self.A, self.b = A, b

    def gradient(self, x, y=None, ctx=None):
        """Compute the gradient of the output (problem solution) with respect to the problem
        parameters. The returned gradient is an ndarray of size (prob.dim_y, prob.dim_x). In
        the case of 1-dimensional parameters the gradient is a vector of size (prob.dim_y,)."""

        # compute optimal value if not already done so
        if y is None:
            y, ctx = self.solve(x)
        assert self._check_constraints(x, y)
        assert self._check_optimality_cond(x, y, ctx)

        # TODO: write val case for LinEqConstDeclarativeNode
        # use cholesky to solve H^{-1}A^T and H^{-1}B
        C, L = sci.linalg.cho_factor(self.fYY(x, y))
        invHAT = sci.linalg.cho_solve((C, L), self.A.T)
        invHB = sci.linalg.cho_solve((C, L), self.fXY(x, y))
        # compute W = H^{-1}A^T (A H^{-1} A^T)^{-1} A
        W = np.dot(invHAT, sci.linalg.solve(np.dot(self.A, invHAT), self.A))
        # return H^{-1}A^T (A H^{-1} A^T)^{-1} A H^{-1} B - H^{-1} B
        return np.dot(W, invHB) - invHB

    def _check_constraints(self, x, y):
        """Check that the problem's constraints are satisfied."""
        residual = np.dot(self.A, y) - self.b
        return np.all(np.abs(residual) <= self.eps)

    def _check_optimality_cond(self, x, y, ctx=None):
        """Checks that the problem's first-order optimality condition is satisfied."""
        warnings.warn("optimality check not implemented yet")
        return True


class NonUniqueDeclarativeNode(AbstractDeclarativeNode):
    """
    A general deep declarative node having non-unique solutions so that the pseudo-inverse is required
    in computing the gradient.
    """
    def __init__(self, n, m):
        super().__init__(n, m)

    def gradient(self, x, y=None, ctx=None):
        """
        Computes the gradient of the output (problem solution) with respect to the problem parameters
        using a pseudo-inverse. The returned gradient is an ndarray of size (self.dim_y, self.dim_x).
        In the case of 1-dimensional parameters the gradient is a vector of size (self.dim_y,).
        """

        # compute optimal value if not already done so
        if y is None:
            y, ctx = self.solve(x)
        assert self._check_optimality_cond(x, y, ctx), abs(self.fY(x, y))

        return -1.0 * np.linalg.pinv(self.fYY(x, y)).dot(self.fXY(x, y))

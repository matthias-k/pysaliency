"""
Author: Matthias Kuemmerer, 2014

Some wrappers around scipy.optimize.minimize to make optimization
of functions with multiple parameters easier
"""

from __future__ import absolute_import, print_function, unicode_literals, division

import numpy as np
import scipy.optimize

from .jacobian import FunctionWithApproxJacobian


class LinearConstraint(object):
    """Allows to build a linear constraint over the parameters by specifying one matrix for each parameter which will be concatenated as needed"""
    def __init__(self, A_dict, lb=-np.inf, ub=np.inf, keep_feasible=False):
        self.A_dict = A_dict
        self.lb = lb
        self.ub = ub
        self.keep_feasible = keep_feasible


class ParameterManager(object):
    def __init__(self, parameters, optimize, **kwargs):
        """ Create a parameter manager
            :param parameters: The parameters to manage
            :type parameters: list of strings
            :param optimize: The parameters that should be optimized. Has to be a subset of parameters
            :type optimize: list of strings
            :param **kwargs: Initial values of the parameters
        """
        self.parameters = parameters
        self.optimize = optimize
        self.param_values = kwargs

    def extract_parameters(self, x, return_list=False):
        """Return dictionary of optimization parameters from vector x.
           The non-optimization parameters will be taken from the initial values.
           if return_list==True, return a list instead of an dictionary"""
        params = self.param_values.copy()
        index = 0
        for param_name in self.optimize:
            if not isinstance(self.param_values[param_name], np.ndarray) or len(self.param_values[param_name].shape) == 0:
                # Only scalar value
                params[param_name] = x[index]
                index += 1
            else:
                shape = self.param_values[param_name].shape
                if len(shape) > 1:
                    raise ValueError('Arrays with more than one dimension are not yet supported!')
                params[param_name] = x[index:index+shape[0]]
                index += shape[0]
        if return_list:
            return [params[key] for key in self.parameters]
        else:
            return params

    def build_vector(self, **kwargs):
        """Build a vector of the optimization parameters.
           The initial values will be taken unless you overwrite
           them using the keyword arguments"""
        params = self.param_values.copy()
        params.update(kwargs)
        vector_values = [params[key] for key in self.optimize]
        return np.hstack(vector_values)

    def get_length(self, param_name):
        """Return the length of parameter param_name when it is used in the optimization vector"""
        if not isinstance(self.param_values[param_name], np.ndarray):
            # Only scalar value
            return 1
        else:
            shape = self.param_values[param_name].shape
            if len(shape) > 1:
                raise ValueError('Arrays with more than one dimension are not yet supported!')
            if len(shape) == 0:
                return 1
            return shape[0]


class KeywordParameterManager(ParameterManager):
    def __init__(self, initial_dict, optimize):
        """ Create a parameter manager
            :param initial_dict: Dictionary of initial values
            :type initial_dict: dict
            :param optimize: The parameters that should be optimized. Has to be a subset of inital_dict.keys()
            :type optimize: list of strings
        """
        parameters = sorted(initial_dict.keys())
        super(KeywordParameterManager, self).__init__(parameters, optimize, **initial_dict)


def wrap_parameter_manager(f, parameter_manager, additional_kwargs=None):
    if isinstance(parameter_manager, KeywordParameterManager):
        def new_f(x, *args, **kwargs):
            if args:
                raise ValueError('KeywordParameterManager can only be used with keyword! Try giving all arguments as keywords.')
            params = parameter_manager.extract_parameters(x, return_list = False)
            kwargs.update(params)
            if additional_kwargs:
                kwargs.update(additional_kwargs)
            return f(**kwargs)
        return new_f
    else:
        def new_f(x, *args, **kwargs):
            params = parameter_manager.extract_parameters(x, return_list = True)
            params.extend(args)
            if additional_kwargs:
                kwargs.update(additional_kwargs)
            return f(*params, **kwargs)
        return new_f



def wrap_linear_constraint(constraint: LinearConstraint, parameter_manager: 'ParameterManager') -> scipy.optimize.LinearConstraint:
    """Wraps a LinearConstraint object into a scipy.optimize.LinearConstraint object"""
    # TODO: hande case that all relevant parameters are not being optimized

    # get output dimension
    output_dims = []
    for A in constraint.A_dict.values():
        if A is not None:
            output_dims.append(A.shape[0])

    if len(set(output_dims)) > 1:
        raise ValueError(f'All A matrices must have the same number of rows! Got {output_dims}')

    output_dim = output_dims[0]

    A_dict = dict(constraint.A_dict)

    lb = constraint.lb
    ub = constraint.ub
    for param_name in parameter_manager.parameters:
        A = constraint.A_dict.get(param_name, None)
        if param_name in parameter_manager.optimize:
            if A is None:
                length = parameter_manager.get_length(param_name)
                A = np.zeros((output_dim, length))
                A_dict[param_name] = A
        else:
            if A is not None:
                param_value = np.atleast_1d(parameter_manager.param_values[param_name])
                constraint_value = np.dot(A, param_value)
                if np.isfinite(lb):
                    lb = lb - constraint_value
                if np.isfinite(ub):
                    ub = ub - constraint_value

    A_list = [A_dict[param_name] for param_name in parameter_manager.optimize]
    A = np.concatenate(A_list, axis=1)
    return scipy.optimize.LinearConstraint(A=A, lb=lb, ub=ub, keep_feasible=constraint.keep_feasible)


def minimize(f, parameter_manager_or_x0, optimize=None, args=(), kwargs=None, method='BFGS',
             jac=None,
             bounds=None,
             constraints=(),
             tol=None,
             options=None,
             jac_approx = FunctionWithApproxJacobian,
             callback=None):
    """Minimize function f with scipy.optimize.minimze, using the parameters
       and initial values from the parameter_manager.

       Remark: Notice that at least SLSQP does not support None values in the bounds"""

    if isinstance(parameter_manager_or_x0, ParameterManager):
        parameter_manager = parameter_manager_or_x0
        if optimize is not None:
            parameter_manager.optimize = optimize
    else:
        parameter_manager = KeywordParameterManager(parameter_manager_or_x0, optimize)
        if args:
            raise ValueError('Keyword based parameters can only be used with kwargs, not with args! Try giving all additional arguments as keywords.')

    wrapped_f = wrap_parameter_manager(f, parameter_manager, kwargs)
    x0 = parameter_manager.build_vector()
    if callable(jac):
        def jac_with_keyword(*args, **kwargs):
            kwargs['optimize'] = parameter_manager.optimize
            ret = jac(*args, **kwargs)
            param_values = parameter_manager.param_values.copy()
            for i, param_name in enumerate(parameter_manager.optimize):
                param_values[param_name] = ret[i]
            return parameter_manager.build_vector(**param_values)
        fun_ = wrapped_f
        jac_ = wrap_parameter_manager(jac_with_keyword, parameter_manager, kwargs)
    elif bool(jac):
        def func_with_keyword(*args, **kwargs):
            kwargs['optimize'] = parameter_manager.optimize
            func_value, jacs = f(*args, **kwargs)
            param_values = parameter_manager.param_values.copy()
            for i, param_name in enumerate(parameter_manager.optimize):
                param_values[param_name] = jacs[i]
            return func_value, parameter_manager.build_vector(**param_values)
        fun_ = wrap_parameter_manager(func_with_keyword, parameter_manager, kwargs)
        jac_ = True
    else:
        fun = jac_approx(wrapped_f, 1e-8)
        jac_ = fun.jac
        fun_ = fun.func

    # Adapt constraints
    if isinstance(constraints, dict):
        constraints = [constraints]
    new_constraints = []
    for constraint in constraints:
        if isinstance(constraint, dict):
            new_constraint = constraint.copy()
            new_constraint['fun'] = wrap_parameter_manager(constraint['fun'], parameter_manager)
            new_constraints.append(new_constraint)
        elif isinstance(constraint, LinearConstraint):
            new_constraints.append(wrap_linear_constraint(constraint, parameter_manager))
        else:
            raise NotImplementedError('Only dictionaries and LinearConstraints are supported as constraints')

    #Adapt bounds:
    if bounds is not None:
        new_bounds = []
        for param_name in parameter_manager.optimize:
            if param_name in bounds:
                new_bounds.extend(bounds[param_name])
            else:
                length = parameter_manager.get_length(param_name)
                for i in range(length):
                    new_bounds.append((-np.inf, np.inf))
        lb, ub = np.array(new_bounds).T
        new_bounds = scipy.optimize.Bounds(lb, ub, keep_feasible=True)
    else:
        new_bounds = None
    if callback is not None:
        callback = wrap_parameter_manager(callback, parameter_manager)

    res = scipy.optimize.minimize(fun_, x0, args=args, jac=jac_,
                                  method=method,
                                  constraints=new_constraints,
                                  bounds=new_bounds,
                                  tol=tol,
                                  callback=callback,
                                  options=options)

    params = parameter_manager.extract_parameters(res.x)
    for key in parameter_manager.parameters:
        setattr(res, key, params[key])
    return res

if __name__ == '__main__':
    def testfunc(x):
        return np.sum(x ** 4)
    testFun = FunctionWithApproxJacobian(testfunc, 1e-8)
    x0 = np.zeros(3)
    #val = testFun(x0)
    #print
    #print val
    g = testFun.jac(x0)
    print()
    print(g)

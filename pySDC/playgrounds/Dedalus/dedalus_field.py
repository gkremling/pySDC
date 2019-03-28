import numpy as np

from dedalus import public as de

from pySDC.core.Errors import DataError


class dedalus_field(object):
    """
    Dedalus data type

    This data type can be used whenever structured data with a single unknown per point in space is required

    Attributes:
        values: contains the domain field
        domain: contains the domain
    """

    def __init__(self, init=None, val=None):
        """
        Initialization routine

        Args:
            init: can either be a tuple (one int per dimension) or a number (if only one dimension is requested)
                  or another mesh object
            val: initial value (default: None)
        Raises:
            DataError: if init is none of the types above
        """

        # if init is another mesh, do a deepcopy (init by copy)
        if isinstance(init, de.Domain):
            self.domain = init
            self.values = init.new_field()
            self.values['g'][:] = val
        elif isinstance(init, type(self)):
            self.domain = init.domain
            self.values = self.domain.new_field()
            self.values['g'] = np.copy(init.values['g'])
        else:
            raise DataError('something went wrong during %s initialization' % type(self))

    def __add__(self, other):
        """
        Overloading the addition operator for mesh types

        Args:
            other (mesh.mesh): mesh object to be added
        Raises:
            DataError: if other is not a mesh object
        Returns:
            mesh.mesh: sum of caller and other values (self+other)
        """

        if isinstance(other, type(self)):
            # always create new mesh, since otherwise c = a + b changes a as well!
            me = dedalus_field(other.domain)
            me.values['g'] = self.values['g'] + other.values['g']
            return me
        else:
            raise DataError("Type error: cannot add %s to %s" % (type(other), type(self)))

    def __sub__(self, other):
        """
        Overloading the subtraction operator for mesh types

        Args:
            other (mesh.mesh): mesh object to be subtracted
        Raises:
            DataError: if other is not a mesh object
        Returns:
            mesh.mesh: differences between caller and other values (self-other)
        """

        if isinstance(other, type(self)):
            # always create new mesh, since otherwise c = a - b changes a as well!
            me = dedalus_field(other.domain)
            me.values['g'] = self.values['g'] - other.values['g']
            return me
        else:
            raise DataError("Type error: cannot subtract %s from %s" % (type(other), type(self)))

    def __rmul__(self, other):
        """
        Overloading the right multiply by factor operator for mesh types

        Args:
            other (float): factor
        Raises:
            DataError: is other is not a float
        Returns:
            mesh.mesh: copy of original values scaled by factor
        """

        if isinstance(other, float):
            # always create new mesh, since otherwise c = a * factor changes a as well!
            me = dedalus_field(self)
            me.values['g'] = other * self.values['g']
            return me
        else:
            raise DataError("Type error: cannot multiply %s to %s" % (type(other), type(self)))

    def __abs__(self):
        """
        Overloading the abs operator for mesh types

        Returns:
            float: absolute maximum of all mesh values
        """

        # take absolute values of the mesh values
        absval = np.linalg.norm(self.values['g'], np.inf)

        return absval

    def apply_mat(self, A):
        """
        Matrix multiplication operator

        Args:
            A: a matrix

        Returns:
            mesh.mesh: component multiplied by the matrix A
        """
        if not A.shape[1] == self.values.shape[0]:
            raise DataError("ERROR: cannot apply operator %s to %s" % (A.shape[1], self))

        me = dedalus_field(self)
        me.values['g'] = A.dot(self.values['g'])

        return me

    # def send(self, dest=None, tag=None, comm=None):
    #     """
    #     Routine for sending data forward in time (blocking)
    #
    #     Args:
    #         dest (int): target rank
    #         tag (int): communication tag
    #         comm: communicator
    #
    #     Returns:
    #         None
    #     """
    #
    #     comm.send(self.values, dest=dest, tag=tag)
    #     return None
    #
    # def isend(self, dest=None, tag=None, comm=None):
    #     """
    #     Routine for sending data forward in time (non-blocking)
    #
    #     Args:
    #         dest (int): target rank
    #         tag (int): communication tag
    #         comm: communicator
    #
    #     Returns:
    #         request handle
    #     """
    #     return comm.isend(self.values, dest=dest, tag=tag)
    #
    # def recv(self, source=None, tag=None, comm=None):
    #     """
    #     Routine for receiving in time
    #
    #     Args:
    #         source (int): source rank
    #         tag (int): communication tag
    #         comm: communicator
    #
    #     Returns:
    #         None
    #     """
    #     self.values = comm.recv(source=source, tag=tag)
    #     return None
    #
    # def bcast(self, root=None, comm=None):
    #     """
    #     Routine for broadcasting values
    #
    #     Args:
    #         root (int): process with value to broadcast
    #         comm: communicator
    #
    #     Returns:
    #         broadcasted values
    #     """
    #     return comm.bcast(self, root=root)


class rhs_imex_dedalus_field(object):
    """
    RHS data type for meshes with implicit and explicit components

    This data type can be used to have RHS with 2 components (here implicit and explicit)

    Attributes:
        impl (mesh.mesh): implicit part
        expl (mesh.mesh): explicit part
    """

    def __init__(self, init, val=0.0):
        """
        Initialization routine

        Args:
            init: can either be a tuple (one int per dimension) or a number (if only one dimension is requested)
                  or another rhs_imex_field object
            val (float): an initial number (default: 0.0)
        Raises:
            DataError: if init is none of the types above
        """

        # if init is another rhs_imex_field, do a deepcopy (init by copy)
        if isinstance(init, type(self)):
            self.domain = init.domain
            self.impl = dedalus_field(init.impl)
            self.expl = dedalus_field(init.expl)
        elif isinstance(init, de.Domain):
            self.domain = init
            self.impl = dedalus_field(init, val=val)
            self.expl = dedalus_field(init, val=val)
        else:
            raise DataError('something went wrong during %s initialization' % type(self))

    def __sub__(self, other):
        """
        Overloading the subtraction operator for rhs types

        Args:
            other (mesh.rhs_imex_field): rhs object to be subtracted
        Raises:
            DataError: if other is not a rhs object
        Returns:
            mesh.rhs_imex_field: differences between caller and other values (self-other)
        """

        if isinstance(other, rhs_imex_dedalus_field):
            # always create new rhs_imex_field, since otherwise c = a - b changes a as well!
            me = rhs_imex_dedalus_field(self.domain)
            me.impl = self.impl - other.impl
            me.expl = self.expl - other.expl
            return me
        else:
            raise DataError("Type error: cannot subtract %s from %s" % (type(other), type(self)))

    def __add__(self, other):
        """
         Overloading the addition operator for rhs types

        Args:
            other (mesh.rhs_imex_field): rhs object to be added
        Raises:
            DataError: if other is not a rhs object
        Returns:
            mesh.rhs_imex_field: sum of caller and other values (self-other)
        """

        if isinstance(other, rhs_imex_dedalus_field):
            # always create new rhs_imex_field, since otherwise c = a + b changes a as well!
            me = rhs_imex_dedalus_field(self.domain)
            me.impl = self.impl + other.impl
            me.expl = self.expl + other.expl
            return me
        else:
            raise DataError("Type error: cannot add %s to %s" % (type(other), type(self)))

    def __rmul__(self, other):
        """
        Overloading the right multiply by factor operator for mesh types

        Args:
            other (float): factor
        Raises:
            DataError: is other is not a float
        Returns:
             mesh.rhs_imex_field: copy of original values scaled by factor
        """

        if isinstance(other, float):
            # always create new rhs_imex_field
            me = rhs_imex_dedalus_field(self.domain)
            me.impl = other * self.impl
            me.expl = other * self.expl
            return me
        else:
            raise DataError("Type error: cannot multiply %s to %s" % (type(other), type(self)))
    #
    # def apply_mat(self, A):
    #     """
    #     Matrix multiplication operator
    #
    #     Args:
    #         A: a matrix
    #
    #     Returns:
    #         mesh.rhs_imex_field: each component multiplied by the matrix A
    #     """
    #
    #     if not A.shape[1] == self.impl.values.shape[0]:
    #         raise DataError("ERROR: cannot apply operator %s to %s" % (A, self.impl))
    #     if not A.shape[1] == self.expl.values.shape[0]:
    #         raise DataError("ERROR: cannot apply operator %s to %s" % (A, self.expl))
    #
    #     me = rhs_imex_dedalus_field(self.domain)
    #     me.impl.values['g'] = A.dot(self.impl.values['g'])
    #     me.expl.values['g'] = A.dot(self.expl.values['g'])
    #
    #     return me

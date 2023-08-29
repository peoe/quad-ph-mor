import numpy as np

from pymor.operators.interface import Operator
from pymor.vectorarrays.block import BlockVectorSpace
from pymor.vectorarrays.numpy import NumpyVectorSpace


class QuadraticInputOperator(Operator):
    linear = True
    block_source = True

    def __init__(self, source, block_source=True):
        source_dim = source.dim
        source_id = source.id

        self.lin_dim = source_dim
        self.lin_range = source
        self.quad_dim = (source_dim * (source_dim + 1)) // 2
        self.quad_range = NumpyVectorSpace(self.quad_dim, id=source_id)

        self.source = source
        if block_source:
            self.range = BlockVectorSpace([ self.lin_range, self.quad_range ])
        else:
            self.range = NumpyVectorSpace(self.lin_dim + self.quad_dim)

    def apply(self, U, mu=None):
        if mu is not None and len(mu) != 0:
            assert 'input' in mu
            input = mu['input']
        else:
            input = np.ones(U.dim)
        ln = len(input)

        return self.range.from_numpy(
            np.concatenate([
                np.eye(ln),
                np.concatenate([
                    np.concatenate([
                        np.zeros(i * ln - (i * (i - 1)) // 2) if i != 0 else [],
                        input[i:],
                        np.zeros((ln - i - 1) * (ln - i - 1) - ((ln - i - 1) * (ln - i - 2)) // 2) if i != ln - 1 else []
                    ]).reshape(1, -1)
                for i in range(ln) ])
            ], axis=1)
        )

    def as_range_array(self, mu=None):
        if mu is not None and len(mu) != 0:
            assert 'input' in mu
            input = mu['input']
        else:
            input = np.ones(self.source.dim)
        ln = len(input)

        return self.range.from_numpy(
            np.concatenate([
                np.eye(ln),
                np.concatenate([
                    np.concatenate([
                        np.zeros(i * ln - (i * (i - 1)) // 2) if i != 0 else [],
                        input[i:],
                        np.zeros((ln - i - 1) * (ln - i - 1) - ((ln - i - 1) * (ln - i - 2)) // 2) if i != ln - 1 else []
                    ]).reshape(1, -1)
                for i in range(ln) ])
            ], axis=1)
        )

    def apply_adjoint(self, V, mu=None):
        V_np = V.to_numpy()
        V_np = V_np[:, :self.source.dim]
        return self.source.from_numpy(V_np)

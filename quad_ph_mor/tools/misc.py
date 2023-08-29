import numpy as np

from itertools import product
from scipy.sparse import coo_matrix
from tqdm import tqdm

from pymor.algorithms.to_matrix import to_matrix
from pymor.models.iosys import LTIModel, PHLTIModel
from pymor.operators.block import BlockDiagonalOperator, BlockRowOperator
from pymor.operators.constructions import LincombOperator, ComponentProjectionOperator, ZeroOperator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.vectorarrays.interface import VectorArray
from pymor.vectorarrays.numpy import NumpyVectorSpace


def to_lti(model):
    # D = model.S - model.N
    # if isinstance(D, LincombOperator):
    #     if all([isinstance(op, ZeroOperator) for op in D.operators]):
    #         D = None
    # elif np.allclose(to_matrix(D), 0.):

    if isinstance(model, PHLTIModel):
        D = (model.S-model.N).assemble()
        D = D.matrix if isinstance(D, NumpyMatrixOperator) else np.zeros((D.range.dim, D.source.dim))
        if np.linalg.norm(D) == 0.:
            D = None
        else:
            D = NumpyMatrixOperator(D)
        return LTIModel(A=model.J-model.R, B=model.G-model.P, C=(model.G + model.P).H, D=D, E=model.E)
    elif isinstance(model, LTIModel):
        D = model.D.assemble()
        D = D.matrix if isinstance(D, NumpyMatrixOperator) else np.zeros((D.range.dim, D.source.dim))
        if np.linalg.norm(D) == 0.:
            D = None
        else:
            D = NumpyMatrixOperator(D)
        return LTIModel(A=model.A, B=model.B, C=model.C, D=model.D, E=model.E)
    else:
        print(f'Encountered unexpected model of type {type(model)}!')
        raise NotImplementedError


def khatri_rao(data, data_2=None, id=None):
    assert isinstance(data, VectorArray)
    if id is None and data.space.id is not None:
        _id = data.space.id + '_QUAD'
    else:
        _id = id
    data_np = data.to_numpy()
    data_2_np = data_np if data_2 is None else data_2.to_numpy()
    assert data_np.shape == data_2_np.shape

    matrices = []
    for i in range(len(data_np)):
        stackable = []
        for j in range(len(data_2_np[i])):
            squared = data_np[i, j] * data_2_np[i, j:]
            stackable.append(squared)
        concat = np.concatenate(stackable)
        matrices.append(concat.reshape(1, len(concat)))
    data_out = np.concatenate(matrices)

    quad_dim = (data.dim * (data.dim + 1)) // 2
    assert data_out.shape[1] == quad_dim
    quad_data_space = NumpyVectorSpace(quad_dim, id=_id)

    return quad_data_space.from_numpy(data_out)


def khatri_rao_np(data, data_2=None, id=None):
    assert isinstance(data, np.ndarray)
    assert data_2 is None or isinstance(data_2, np.ndarray)

    space = NumpyVectorSpace(dim=data.shape[1], id=id)

    return khatri_rao(space.from_numpy(data), data_2=None if data_2 is None else space.from_numpy(data_2), id=id).to_numpy()


def kron(data, data_2=None, id=None):
    assert isinstance(data, VectorArray)
    if id is None and data.space.id is not None:
        _id = data.space.id + '_QUAD'
    else:
        _id = id
    data_np = data.to_numpy()
    data_2_np = data_np if data_2 is None else data_2.to_numpy()
    assert data_np.shape == data_2_np.shape

    matrices = []
    for i in range(data_np.shape[0]):
        stackable = []
        for j in range(data_2_np.shape[1]):
            squared = data_np[i, j] * data_2_np[i:, j:]
            stackable.append(squared)
        concat = np.concatenate(stackable, axis=1)
        matrices.append(concat)
    data_out = np.concatenate(matrices)

    quad_dim = (data.dim * (data.dim + 1)) // 2
    assert data_out.shape == (quad_dim, quad_dim)
    quad_data_space = NumpyVectorSpace(quad_dim, id=_id)

    return quad_data_space.from_numpy(data_out)


def kron_np(data, data_2=None, id=None):
    assert isinstance(data, np.ndarray)
    assert data_2 is None or isinstance(data_2, np.ndarray)

    space = NumpyVectorSpace(dim=data.shape[1], id=id)

    return kron(space.from_numpy(data), data_2=None if data_2 is None else space.from_numpy(data_2), id=id).to_numpy()


def conj(vec):
    return np.conj(vec.T)


def vec2mat(vector, dims):
    assert isinstance(dims, tuple)
    assert len(dims) == 2
    vec = vector.reshape(-1)
    assert vec.shape[0] == dims[0] * dims[1]

    row_inds = []
    col_inds = []
    for i in range(dims[0]):
        row_inds.append(np.repeat(i, dims[1]))
        col_inds.append(np.array(range(dims[1])))
    row_inds = np.concatenate(row_inds)
    col_inds = np.concatenate(col_inds)

    matrix = coo_matrix((vec, (row_inds, col_inds)), dims)
    matrix.eliminate_zeros()

    return matrix


def vec2upper(vector, dim):
    assert isinstance(dim, int)
    vec = vector.reshape(-1)
    assert vec.shape[0] == (dim * (dim + 1)) // 2

    row_inds = []
    col_inds = []
    for i in range(dim):
        row_inds.append(np.repeat(i, dim - i))
        col_inds.append(np.array(range(i, dim)))
    row_inds = np.concatenate(row_inds)
    col_inds = np.concatenate(col_inds)

    matrix = coo_matrix((vec, (row_inds, col_inds)), (dim, dim))
    matrix.eliminate_zeros()

    return matrix


def vec2strict(vector, dim):
    assert isinstance(dim, int)
    vec = vector.reshape(-1)
    assert vec.shape[0] == (dim * (dim - 1)) // 2

    row_inds = []
    col_inds = []
    for i in range(dim):
        row_inds.append(np.repeat(i, dim - 1 - i))
        col_inds.append(np.array(range(i + 1, dim)))
    row_inds = np.concatenate(row_inds)
    col_inds = np.concatenate(col_inds)

    matrix = coo_matrix((vec, (row_inds, col_inds)), (dim, dim))
    matrix.eliminate_zeros()

    return matrix


def mat2vec(matrix):
    assert len(matrix.shape) == 2

    dims = matrix.shape
    vec = np.zeros(dims[0] * dims[1], dtype=np.complex128)

    for i in range(dims[0]):
        if isinstance(matrix, np.ndarray):
            vec[i * dims[1]:(i + 1) * dims[1]] = matrix[i, :dims[1]]
        else:
            vec[i * dims[1]:(i + 1) * dims[1]] = matrix.tocsc()[i, :dims[1]].toarray()

    return vec


def upper2vec(matrix):
    assert len(matrix.shape) == 2
    assert matrix.shape[0] == matrix.shape[1]

    dim = matrix.shape[0]
    vec = np.zeros((dim * (dim + 1)) // 2, dtype=np.complex128)

    for i in range(dim):
        pre_index = i * (dim + 1) - ((i * (i + 1)) // 2)
        post_index = (i + 1) * dim - ((i * (i + 1)) // 2)
        if isinstance(matrix, np.ndarray):
            vec[pre_index:post_index] = matrix[i, i:dim]
        else:
            vec[pre_index:post_index] = matrix.tocsc()[i, i:dim].toarray()

    return vec


def strict2vec(matrix):
    assert len(matrix.shape) == 2
    assert matrix.shape[0] == matrix.shape[1]

    dim = matrix.shape[0]
    vec = np.zeros((dim * (dim - 1)) // 2, dtype=np.complex128)

    for i in range(dim):
        pre_index = i * dim - (i * (i + 1)) // 2
        post_index = (i + 1) * dim - ((i + 1) * (i + 2)) // 2
        if isinstance(matrix, np.ndarray):
            vec[pre_index:post_index] = matrix[i, i + 1:dim]
        else:
            vec[pre_index:post_index] = matrix.tocsc()[i, i + 1:dim].toarray()

    return vec


def _measure(matrices, sample_points, treatments, disable_tqdm):
    tfs = []
    for mats, treatment in zip(matrices, treatments):
        model = LTIModel.from_matrices(*mats)
        tfs.append((model.transfer_function.tf, treatment))

    measurements = []
    for sample_point in tqdm(sample_points, desc='Measuring TF', total=len(sample_points), unit_scale=True, disable=disable_tqdm):
        values = []
        for tf, treatment in tfs:
            values.append(tf(sample_point) if treatment is None else treatment(tf(sample_point)))
        measurements.append((sample_point, values))

    return measurements


def measure_tfs(model_matrices, sample_points, treatments=None, disable_tqdm=True):
    assert isinstance(model_matrices, list)
    assert np.all([ isinstance(m, tuple) and len(m) == 5 for m in model_matrices ])

    assert treatments is None or (isinstance(treatments, list) and np.all([treatment in (kron_np, None) for treatment in treatments]))
    if treatments is None:
        treatments = [ None for _ in model_matrices ]
    assert len(treatments) == len(model_matrices)

    measurements = []

    measurements = _measure(model_matrices, sample_points=sample_points, treatments=treatments, disable_tqdm=disable_tqdm)

    return measurements


def greedy_update(sample_points, target_mats, current_mats):
    model = LTIModel.from_matrices(*target_mats)
    current_model = LTIModel.from_matrices(*current_mats)
    _, fpeak = (model - current_model).hinf_norm(return_fpeak=True)
    return np.concatenate([sample_points, np.array([1j * fpeak])]), np.array([1j * fpeak])


def update_sample_points(sample_points, gamma, target_mats, current_mats, rom_mats=None, search_depth=5):
    def _error(point):
        mats = [target_mats, current_mats] if rom_mats is None else [target_mats, current_mats, rom_mats]
        meas = measure_tfs(mats, sample_points=[1j * point])[0][1]
        target_val = meas[0] if rom_mats is None else kron_np(meas[0] - meas[2])
        current_val = meas[1]
        return np.linalg.norm(target_val - current_val, ord=2)

    prev_num_samples = len(sample_points)

    n_new = 1
    current_depth = 0
    new_points = []
    while n_new > 0 and current_depth < search_depth:
        point_imags = np.sort(np.imag(sample_points))
        assert np.all(point_imags >= 0.)

        # populate error dict
        errors = {}
        for index in range(len(sample_points) - 1):
            if index not in errors:
                errors[index] = _error(point_imags[index])
            errors[index + 1] = _error(point_imags[index + 1])

        # check if errors are any good
        n_new = 0
        point_next = point_imags[0]
        error_next = errors[0]
        for index in range(len(sample_points) - 1):
            point_prev = point_next if point_next != 0. else 1e-15
            error_prev = error_next
            point_next = point_imags[index + 1]
            error_next = errors[index + 1]

            new_candidate = 10**(.5 * (np.log10(point_prev) + np.log10(point_next)))
            d1 = (_error(new_candidate) - error_prev) / (new_candidate - point_prev) if new_candidate != point_prev else 0.
            d2 = (error_next - _error(new_candidate)) / (point_next - new_candidate) if new_candidate != point_next else 0.
            gamma_candidate = max([error_prev, error_next])
            d = max([d1, d2])

            if not(new_candidate == point_prev and new_candidate == point_next):
                if d * (point_next - point_prev) >= 2 * (gamma_candidate + gamma) - (error_next + error_prev):
                    sample_points = np.concatenate([sample_points, np.array([1j * new_candidate])])
                    new_points.append(1j * new_candidate)
                    n_new += 1

        current_depth += 1

    if len(sample_points) > prev_num_samples:
        print(f'Updated to {len(sample_points)} sample points!')

    return sample_points, np.array(new_points)


def add_models(left, right):
    assert ((isinstance(left, PHLTIModel) and isinstance(right, PHLTIModel)) or
            (isinstance(left, LTIModel) and isinstance(right, LTIModel)))

    if isinstance(left, PHLTIModel) and isinstance(right, PHLTIModel):
        assert left.E.source == right.E.source
        assert left.E.range == right.E.range
        assert left.J.source == right.J.source
        assert left.J.range == right.J.range
        assert left.R.source == right.R.source
        assert left.R.range == right.R.range

        assert left.G.range == right.G.range
        assert left.P.range == right.P.range

        J = to_matrix(left.J + right.J)
        R = to_matrix(left.R + right.R)
        E = to_matrix(left.E + right.E)

        left_G = to_matrix(left.G)
        right_G = to_matrix(right.G)
        left_P = to_matrix(left.P)
        right_P = to_matrix(right.P)

        G = np.concatenate([
            left_G if isinstance(left_G, np.ndarray) else left_G.toarray(),
            right_G if isinstance(right_G, np.ndarray) else right_G.toarray()
        ], axis=1)
        P = np.concatenate([
            left_P if isinstance(left_P, np.ndarray) else left_P.toarray(),
            right_P if isinstance(right_P, np.ndarray) else right_P.toarray()
        ], axis=1)

        if left.N is not None and right.N is not None:
            left_N = to_matrix(left.N, format='dense')
            right_N = to_matrix(right.N, format='dense')

            if np.allclose(left_N, 0.) and np.allclose(right_N, 0.):
                N = None
            else:
                N = np.concatenate([
                    np.concatenate([
                        left_N, np.zeros((left_N.shape[0], right_N.shape[1]))
                    ], axis=0),
                    np.concatenate([
                        np.zeros((right_N.shape[0], left_N.shape[1])), right_N
                    ], axis=0)
                ], axis=1)
        else:
            N = None

        if left.S is not None and right.S is not None:
            left_S = to_matrix(left.S, format='dense')
            right_S = to_matrix(right.S, format='dense')

            if np.allclose(left_S, 0.) and np.allclose(right_S, 0.):
                S = None
            else:
                S = np.concatenate([
                    np.concatenate([
                        left_S, np.zeros((left_S.shape[0], right_S.shape[1]))
                    ], axis=0),
                    np.concatenate([
                        np.zeros((right_S.shape[0], left_S.shape[1])), right_S
                    ], axis=0)
                ], axis=1)
        else:
            S = None

        return PHLTIModel.from_matrices(E=E, J=J, R=R, G=G, P=P, N=N, S=S, state_id=None)
    else:
        assert left.E.source == right.E.source
        assert left.E.range == right.E.range
        assert left.A.source == right.A.source
        assert left.A.range == right.A.range

        assert left.B.range == right.B.range
        assert left.C.source == right.C.source

        A = to_matrix(left.A + right.A)
        E = to_matrix(left.E + right.E)

        left_B = to_matrix(left.B)
        right_B = to_matrix(right.B)
        left_C = to_matrix(left.C)
        right_C = to_matrix(right.C)

        B = np.concatenate([
            left_B if isinstance(left_B, np.ndarray) else left_B.toarray(),
            right_B if isinstance(right_B, np.ndarray) else right_B.toarray()
        ], axis=1)
        C = np.concatenate([
            left_C if isinstance(left_C, np.ndarray) else left_C.toarray(),
            right_C if isinstance(right_C, np.ndarray) else right_C.toarray()
        ], axis=0)

        if not isinstance(left.D, ZeroOperator) and not isinstance(right.D, ZeroOperator):
            raise NotImplementedError

        return LTIModel.from_matrices(A=A, B=B, C=C, E=E, state_id=None)


def get_dune_boundary_dofs(form):
    bnd_dofs = []
    for i, outer_block in enumerate(form.dirichletBlocks):
        for j, block in enumerate(outer_block):
            if block > 0:
                bnd_dofs.append(i * len(outer_block) + j)

    return bnd_dofs


def product_dict(**kwargs):
    keys = kwargs.keys()
    for instance in product(*kwargs.values()):
        yield dict(zip(keys, instance))


def couple_models(rom, inf):
    E = rom.E + inf.E
    A = rom.A + inf.A
    B = BlockRowOperator([rom.B, inf.B])
    C = BlockRowOperator([rom.C, inf.C])
    D = BlockDiagonalOperator([rom.D, inf.D])
    return LTIModel(A=A, B=B, C=C, D=D, E=E)

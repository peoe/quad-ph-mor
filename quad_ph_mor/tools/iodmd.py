import numpy as np

from pymor.models.iosys import LTIModel


def iodmd(X, Y, U, Xdot=None, E=None, rcond=1e-12):
    assert isinstance(X, np.ndarray)
    assert isinstance(Y, np.ndarray)
    assert isinstance(U, np.ndarray)
    assert X.shape[1] == Y.shape[1]
    assert U.shape == Y.shape

    if Xdot is None:
        Xdot = X[:, 1:]
        X = X[:, :-1]
        Y = Y[:, :-1]
        U = U[:, :-1]

    state_dim = X.shape[0]
    io_dim = U.shape[0]

    if E is None:
        E = np.eye(state_dim)
    else:
        assert isinstance(E, np.ndarray)
        assert E.shape[0] == E.shape[1] == X.shape[0]

    T = np.concatenate([ X, U ])
    Z = np.concatenate([ E @ Xdot, Y ])

    mats = Z @ np.linalg.pinv(T, rcond=rcond)

    A = mats[:state_dim, :state_dim]
    B = mats[:state_dim, state_dim:]
    C = mats[state_dim:, :state_dim]
    D = mats[state_dim:, state_dim:]

    return LTIModel.from_matrices(A=A, B=B, C=C, D=D, E=E), {'abs': np.linalg.norm(Z - mats @ T), 'rel': np.linalg.norm(Z - mats @ T) / np.linalg.norm(Z)}


def oi(X, Y, U, dt, E=None, rcond=1e-12):
    Xdot = 1. / dt * (X[:, 1:] - X[:, :-1])
    X = .5 * (X[:, 1:] + X[:, :-1])
    U = .5 * (U[:, 1:] + U[:, :-1])
    Y = .5 * (Y[:, 1:] + Y[:, :-1])

    return iodmd(X, Y, U, Xdot=Xdot, E=E, rcond=rcond)

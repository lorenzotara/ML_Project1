import numpy as np


def compute_mse(y, tx, w):
    """compute the loss by mse."""

    e = y - tx.dot(w)
    mse = e.dot(e) / (2 * len(e))
    return mse


def compute_gradient(y, tx, w):
    """Compute the gradient."""

    e = y - tx.dot(w)
    gradient = - (1 / len(y)) * tx.T.dot(e)
    return gradient, compute_mse(y, tx, w)


def linear_regression(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w

    for n_iter in range(max_iters):

        gradient, loss = compute_gradient(y, tx, w)

        w = w - gamma * gradient
        ws.append(w)
        losses.append(loss)

        print("Gradient Descent({bi}/{ti}): loss={l}".format(
            bi=n_iter, ti=max_iters - 1, l=loss))

    return losses, w


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""

    pol = np.ones(len(x))

    for n in range(degree):
        pol = np.c_[pol, pow(x, n + 1)]

    return pol


def split_data(x, y, ratio, seed=1):
    """split the dataset based on the split ratio."""
    # set seed
    np.random.seed(seed)

    n_rows_split = int(np.floor(ratio * len(x)))

    x_train = x[:n_rows_split]
    y_train = y[:n_rows_split]

    X_test = x[n_rows_split:]
    Y_test = y[n_rows_split:]

    return x_train, y_train, X_test, Y_test


def cross_validation(y, x, k_indices, k, lambda_, degree):
    """return the loss of ridge regression."""

    test_ind = k_indices[k]
    train_ind = k_indices
    train_ind = np.delete(train_ind, k, axis=0)

    n_fold = len(k_indices)
    new_shape = int(len(x) * (n_fold - 1) / (n_fold) - 1)

    train_ind = np.reshape(train_ind, new_shape)

    x = build_poly(x, degree)

    x_train = x[train_ind]
    x_test = x[test_ind]

    y_train = y[train_ind]
    y_test = y[test_ind]

    loss, ws = ridge_regression(y_train, x_train, lambda_)

    loss_tr = compute_mse(y_train, x_train, ws)
    loss_te = compute_mse(y_test, x_test, ws)

    return loss_tr, loss_te


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)


def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""

    N = len(tx.T.dot(tx))
    lambda_first = 2 * N * lambda_

    a = tx.T.dot(tx) + lambda_first * np.identity(N)
    b = tx.T.dot(y)

    w_ridge = np.linalg.solve(a, b)

    loss = compute_mse(y, tx, w_ridge) + lambda_ * np.linalg.norm(w_ridge)

    return loss, w_ridge


def least_squares(y, tx):
    """calculate the least squares."""

    a = tx.T.dot(tx)
    b = tx.T.dot(y)

    opt_w = np.linalg.solve(a, b)

    return opt_w


def sigmoid(t):
    """apply sigmoid function on t."""

    return 1 / (1 + np.exp(-t))


def lr_compute_loss(y, tx, w):
    """compute the cost by negative log likelihood."""

    sig = sigmoid(tx.dot(w))

    loss = - np.sum(y * np.log(sig) + (1 - y) * np.log(1 - sig))

    return loss


def lr_compute_gradient(y, tx, w):
    """compute the gradient of loss."""
    sig = sigmoid(tx.dot(w))

    return tx.T.dot(sig - y)


def lr_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descent using logistic regression.
    Return the loss and the updated w.
    """
    loss = lr_compute_loss(y, tx, w)

    gradient = lr_compute_gradient(y, tx, w)

    w -= gamma * gradient

    return loss, w


def hessian(tx, w):
    """return the hessian of the loss function."""

    sig = sigmoid(tx.dot(w))
    S = np.identity(len(sig)) * (sig * (1 - sig))
    H = tx.T.dot(S.dot(tx))

    return H


def logistic_regression(y, tx, w):
    """return the loss, gradient, and hessian."""

    return lr_compute_loss(y, tx, w), lr_compute_gradient(y, tx, w), hessian(tx, w)


def newton_method(y, tx, w):
    """
    Do one step on Newton's method.
    return the loss and updated w.
    """

    loss, gradient, H = logistic_regression(y, tx, w)

    a = H
    b = H.dot(w) - gradient
    w = np.linalg.solve(a, b)

    return loss, w


def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss, gradient, and hessian."""

    loss = lr_compute_loss(y, tx, w) + lambda_ / 2 * (np.linalg.norm(w)) ** 2
    gradient = lr_compute_gradient(y, tx, w) + lambda_ * w
    H = hessian(tx, w)

    return loss, gradient, H


def penalized_gradient(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.
    """

    loss, gradient, H = penalized_logistic_regression(y, tx, w, lambda_)

    a = H
    b = H.dot(w) - gamma * gradient
    w = np.linalg.solve(a, b)

    return loss, w




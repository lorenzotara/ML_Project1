import numpy as np
from matplotlib import pyplot as plt
from helpers.data_analysis import *
from helpers.proj1_helpers import *


def compute_mse(y, tx, w):
    """compute the loss by mse."""

    e = y - tx.dot(w)
    mse = e.dot(e) / (2 * len(e))
    return mse


def compute_gradient(y, tx, w):
    """Compute the gradient."""

    e = y - tx.dot(w)
    loss = e.dot(e) / (2 * len(e))
    gradient = - (1 / len(y)) * tx.T.dot(e)
    return gradient, loss


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def mini_batch_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm."""

    ws = [initial_w]
    losses = []
    w = initial_w

    for n_iter in range(max_iters):

        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            gradient, loss = compute_gradient(minibatch_y, minibatch_tx, w)

            w = w - gamma * gradient
            ws.append(w)
            losses.append(loss)

    return losses, ws


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


def add_column_of_ones(x):

    ones = np.ones(len(x))

    return np.c_[ones, x]


def build_poly_with_ones(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""

    pol = np.ones(len(x))

    for n in range(degree):
        pol = np.c_[pol, pow(x, n + 1)]

    return pol


def build_poly(x, degree):

    pol = x

    for n in range(1, degree):
        pol = np.c_[pol, pow(x, n+1)]

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


def cross_validation_visualization(lambds, mse_tr, mse_te):
    """visualization the curves of mse_tr and mse_te."""
    plt.semilogx(lambds, mse_tr, marker=".", color='b', label='train error')
    plt.semilogx(lambds, mse_te, marker=".", color='r', label='test error')
    plt.xlabel("lambda")
    plt.ylabel("rmse")
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("cross_validation")


def lr_cross_validation(y, x, k_indices, k, degree):
    """return the loss of ridge regression."""

    test_ind = k_indices[k]
    train_ind = k_indices
    train_ind = np.delete(train_ind, k, axis=0)

    n_fold = len(k_indices)
    new_shape = int(len(x) * (n_fold - 1) / (n_fold) - 1) + 1

    train_ind = np.reshape(train_ind, new_shape)

    x = build_poly_with_ones(x, degree)
    x[:, 1:len(x)] = features_standardization(x[:, 1:len(x)])

    x_train = x[train_ind]
    x_test = x[test_ind]

    y_train = y[train_ind]
    y_test = y[test_ind]

    loss, ws = least_squares(y_train, x_train)

    loss_tr = compute_mse(y_train, x_train, ws)
    loss_te = compute_mse(y_test, x_test, ws)
    print("\n\n\n")
    print("loss_tr: ", loss_tr)
    print("loss_te: ", loss_te)

    y_pred = predict_labels(ws, x_test)
    final_result = y_test == y_pred

    score = np.count_nonzero(final_result) / len(final_result)

    print("\n", score * 100, "%")

    return loss_tr, loss_te


def lr_ridge_reg_cross_validation(y, x, k_indices, k, lambda_):
    """return the loss of ridge regression."""

    test_ind = k_indices[k]
    train_ind = k_indices
    train_ind = np.delete(train_ind, k, axis=0)

    n_fold = len(k_indices)
    new_shape = int(len(x) * (n_fold - 1) / (n_fold) - 1) + 1

    train_ind = np.reshape(train_ind, new_shape)

    # x = build_poly_with_ones(x, degree)
    # x[:, 1:len(x)] = features_standardization(x[:, 1:len(x)])

    x_train = x[train_ind]
    x_test = x[test_ind]

    y_train = y[train_ind]
    y_test = y[test_ind]

    loss, ws = ridge_regression(y_train, x_train, lambda_)

    loss_tr = compute_mse(y_train, x_train, ws)
    loss_te = compute_mse(y_test, x_test, ws)

    print("loss_tr: ", loss_tr)
    print("loss_te: ", loss_te)

    y_pred = predict_labels(ws, x_test)
    final_result = y_test == y_pred

    score = np.count_nonzero(final_result) / len(final_result)

    print("\n", score * 100, "%")

    return loss_tr, loss_te


def cross_validation(y, x, k_indices, k, lambda_, degree):
    """return the loss of ridge regression."""

    test_ind = k_indices[k]
    train_ind = k_indices
    train_ind = np.delete(train_ind, k, axis=0)

    n_fold = len(k_indices)
    new_shape = int(len(x) * (n_fold - 1) / (n_fold) - 1) + 1

    train_ind = np.reshape(train_ind, new_shape)

    x = build_poly_with_ones(x, degree)
    x[:, 1:len(x)] = features_standardization(x[:, 1:len(x)])

    x_train = x[train_ind]
    x_test = x[test_ind]

    y_train = y[train_ind]
    y_test = y[test_ind]

    loss, ws = least_squares(y_train, x_train, lambda_)

    loss_tr = compute_mse(y_train, x_train, ws)
    loss_te = compute_mse(y_test, x_test, ws)

    return loss_tr, loss_te


def lr_ridge_cross_validation_demo(y, x, k_fold_):
    seed = 1
    k_fold = k_fold_
    lambdas = np.logspace(-4, 0, 30)
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []

    for ind, lambda_ in enumerate(lambdas):

        cross_tr = []
        cross_te = []

        for k in range(k_fold):
            loss_tr, loss_te = lr_ridge_reg_cross_validation(y, x, k_indices, k, lambda_)
            cross_tr.append(loss_tr)
            cross_te.append(loss_te)

        rmse_tr.append(np.sqrt(2 * np.mean(cross_tr)))
        rmse_te.append(np.sqrt(2 * np.mean(cross_te)))

    # cross_validation_visualization(lambdas, rmse_tr, rmse_te)
    return rmse_tr, rmse_te, lambdas


def lr_ridge_cross_val_demo_lambda_fixed(y, x, k_fold_, lambda_):
    seed = 1
    k_fold = k_fold_
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    cross_tr = []
    cross_te = []

    print("\n\n\n")
    print("lambda: ", lambda_)

    for k in range(k_fold):
        loss_tr, loss_te = lr_ridge_reg_cross_validation(y, x, k_indices, k, lambda_)
        cross_tr.append(np.sqrt(2*loss_tr))
        cross_te.append(np.sqrt(2*loss_te))

    # cross_validation_visualization(lambdas, rmse_tr, rmse_te)
    return cross_tr, cross_te


def cross_validation_demo(y, x, degree_, k_fold_):
    seed = 1
    degree = degree_
    k_fold = k_fold_
    lambdas = np.logspace(-4, 0, 30)
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []

    for ind, lambda_ in enumerate(lambdas):

        cross_tr = []
        cross_te = []

        for k in range(k_fold):
            loss_tr, loss_te = cross_validation(y, x, k_indices, k, lambda_, degree)
            cross_tr.append(loss_tr)
            cross_te.append(loss_te)

        rmse_tr.append(np.sqrt(2 * np.mean(cross_tr)))
        rmse_te.append(np.sqrt(2 * np.mean(cross_te)))

    cross_validation_visualization(lambdas, rmse_tr, rmse_te)


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

    loss = compute_mse(y, tx, opt_w)

    print("Least Squares: loss={l}, w={weight}".format(l=loss, weight=opt_w))

    return loss, opt_w


def sigmoid(t):
    """apply sigmoid function on t."""

    # If you need to use the np.linalg.solve you can't use dtype =
    # return 1 / (1 + np.exp(-t, dtype=np.float128))
    return 1 / (1 + np.exp(-t))


def lr_compute_loss(y, tx, w):
    """compute the cost by negative log likelihood."""

    # sig = sigmoid(tx.dot(w))
    # loss = - np.sum(y * np.log(sig) + (1 - y) * np.log(1 - sig))
    #
    # Equivalent

    pred = tx.dot(w)
    loss = np.sum(np.log(1 + np.exp(pred)) - y.T.dot(pred))

    return loss


def lr_compute_gradient(y, tx, w):
    """compute the gradient of loss."""

    pred = tx.dot(w)
    sig = sigmoid(pred)
    gradient = tx.T.dot(sig - y) / len(y)
    loss = - np.sum(y * np.log(sig) + (1 - y) * np.log(1 - sig)) / len(y)
    # loss = (np.sum(np.log(1 + np.exp(pred))) - y.T.dot(pred)) / len(y)

    return loss, gradient


# def lr_gradient_descent(y, tx, w, gamma):
#     """
#     Do one step of gradient descent using logistic regression.
#     Return the loss and the updated w.
#     """
#     # loss = lr_compute_loss(y, tx, w)
#
#     gradient, loss = lr_compute_gradient(y, tx, w)
#
#     w -= gamma * gradient
#
#     return loss, w


def hessian(tx, w):
    """return the hessian of the loss function."""

    sig = sigmoid(tx.dot(w))
    S = np.identity(len(sig)) * (sig * (1 - sig))
    H = tx.T.dot(S.dot(tx))

    return H


def lr_loss_gradient_hessian(y, tx, w):
    """return the loss, gradient, and hessian."""
    loss, gradient = lr_compute_gradient(y, tx, w)
    # print(loss)

    return lr_compute_loss(y, tx, w), gradient, hessian(tx, w)


def logistic_regression2(y, tx, initial_w, max_iters, gamma):

    ws = [initial_w]
    losses = []
    w = initial_w

    for n_iter in range(max_iters):
        loss, gradient = lr_compute_gradient(y, tx, w)

        w -= gamma * gradient
        ws.append(w)
        losses.append(loss)

        print("Logistic Gradient Descent({bi}/{ti}): loss={l}".format(
            bi=n_iter, ti=max_iters - 1, l=loss))

    return losses, w


def newton_method(y, tx, w):
    """
    Do one step on Newton's method.
    return the loss and updated w.
    """

    loss, gradient, H = lr_loss_gradient_hessian(y, tx, w)

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




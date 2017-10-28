import numpy as np
from helpers.proj1_functions import build_poly, batch_iter
from helpers.proj1_functions import build_k_indices
from helpers.proj1_functions import predict_labels_for_lr

#signal clipped in order to not get an overflow
def sigmoid(t):
    """apply sigmoid function on t."""

    #t = np.array(t)
    #return np.where(t < 0, np.exp(t) / (1 + np.exp(t)), 1 / (1 + np.exp(-t)))

    signal = np.clip(t, -500, 500)

    # Calculate activation signal
    signal = 1.0 / (1 + np.exp(-signal))

    signal = np.where(signal == 1, 0.999999999999999, signal)

    signal = np.where(signal == 0, 0.000000000000001, signal)

    return signal

    # If you need to use the np.linalg.solve you can't use dtype =
    #return 1 / (1 + np.exp(-t, dtype=np.float128))
    #return 1 / (1 + np.exp(-t))



def calculate_logistic_loss(y, tx, w):
    """compute the cost by negative log likelihood."""

    sig = sigmoid(tx.dot(w))
    loss = (- np.sum(y * np.log(sig) + (1 - y) * np.log(1 - sig))) / len(y)
    #
    # Equivalent
    #pred = tx.dot(w)
    #loss = np.sum(np.log(1 + np.exp(pred)) - y.T.dot(pred)) / len(y)

    return loss

#compute the gradient of loss.
def calculate_logistic_gradient(y, tx, w):

    sigmoid_res = sigmoid(tx.dot(w))
    loss = calculate_logistic_loss(y, tx, w)
    return loss, tx.T.dot(sigmoid_res - y) / len(y)

"""
Do one step of gradient descent using logistic regression.
Return the loss and the updated w.
"""
def learning_logistic_by_gradient_descent(y, tx, w, gamma):

    # compute the gradient
    loss, gradient = calculate_logistic_gradient(y, tx, w)

    # update w
    w = w - gamma * gradient
    return loss, w

# in the penalized cost we do not consider the first element of the w vector (see implementation Andrew Ng)
def calculate_logistic_penalized_loss(y, tx, w, lambda_):

    #penalized_w = w[1:]
    penalized_w = w
    return calculate_logistic_loss(y, tx, w) + (lambda_ /(2 * len(y))) * np.linalg.norm(penalized_w)


# logistic regression penalized gradient
def calculate_logistic_penalized_gradient(y, tx, w, lambda_):

    loss = calculate_logistic_penalized_loss(y, tx, w, lambda_)

    #penalized_w = np.array(w)
    #penalized_w[0] = 0
    penalized_w = w

    useless, gradient = calculate_logistic_gradient(y, tx, w)
    gradient = gradient + (lambda_ / len(y)) * penalized_w

    return loss, gradient

#one iteration of gradient descent logistic penalized
def penalized_logistic_gradient_descent_iteration(y, tx, w, gamma, lambda_):

    loss, gradient = calculate_logistic_penalized_gradient(y, tx, w, lambda_)
    w = w - gamma * gradient

    return loss, w


#complete gradient descent with penalized logistic regression
def learning_with_penalized_logistic_gradient_desc(y, tx, w, gamma, lambda_, max_iter):

    threshold = 1e-8
    losses = []

    # start the logistic regression
    for iteration in range(max_iter):
        # get loss and update w.

        loss, w = penalized_logistic_gradient_descent_iteration(y, tx, w, gamma, lambda_)
        # log info
        print("Logistic Gradient Descent({bi}/{ti}): loss={l}".format(bi=iteration, ti=max_iter - 1, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break


    return loss, w

# K'th iteration gradient descent penalized logistic
def cross_validation_k_iteration_penalized_logistic(y, x, w, k_indices, k, gamma, lambda_, max_iter, is_stochastic):
    """return the loss of ridge regression."""

    # get k'th subgroup in test, others in train
    k_group_indices = k_indices[k]

    x_test = x[k_group_indices]
    y_test = y[k_group_indices]
    train_indices = np.delete(np.arange(x.shape[0]), k_group_indices)
    x_train = x[train_indices]
    y_train = y[train_indices]

    w = np.random.rand(x_train.shape[1])
    if is_stochastic:
        loss, weights = penalized_logistic_mini_batch_gradient_descent(y_train, x_train, w, 1, max_iter, gamma, lambda_)
    else:
        loss, weights = learning_with_penalized_logistic_gradient_desc(y_train, x_train, w, gamma, lambda_, max_iter)

    # calculate the loss for train and test data
    loss_tr = (calculate_logistic_penalized_loss(y_train, x_train, weights, lambda_))
    loss_te = (calculate_logistic_penalized_loss(y_test, x_test, weights, lambda_))

    print("loss_tr: ", loss_tr)
    print("loss_te: ", loss_te)

    y_pred = predict_labels_for_lr(weights, x_test)
    y_test2 = y_test[np.argwhere(y_test == 0)] = - 1
    final_result = y_test2 == y_pred

    score = np.count_nonzero(final_result) / len(final_result)

    print("\n", score * 100, "%")

    return loss_tr, loss_te, weights


# complete cross validation gradient descent penalized logistic
def cross_validation_penalized_logistic(y, x, w, gamma, k_fold, lambda_, max_iter, is_stochastic):

    seed = 1

    k_indices = build_k_indices(y, k_fold, seed)

    loss_tr_vector = []
    loss_te_vector = []
    w_vector = []
    temp_w = np.zeros(len(w))

    print("\n\n\n")
    print("lambda: ", lambda_)


    for k in range(k_fold):

        loss_tr, loss_te, w = cross_validation_k_iteration_penalized_logistic(y, x, w, k_indices, k, gamma, lambda_, max_iter, is_stochastic)
        loss_tr_vector.append(loss_tr)
        loss_te_vector.append(loss_te)
        temp_w = temp_w + w

    loss_tr = np.average(loss_tr_vector)
    loss_te = np.average(loss_te_vector)
    ws = temp_w / k_fold


    return loss_tr, loss_te, ws


def penalized_logistic_mini_batch_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma, lambda_):
    ws = [initial_w]
    losses = []
    w = initial_w

    for n_iter in range(max_iters):

        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            gradient, loss = calculate_logistic_penalized_gradient(minibatch_y, minibatch_tx, w, lambda_)

            w = w - gamma * gradient
            ws.append(w)
            losses.append(loss)

    return losses, w


#####NEWTON METHOD####

def calculate_hessian(tx, w):
    """return the hessian of the loss function."""

    sig = sigmoid(tx.dot(w))
    neg_sig = 1 - sig
    diag = sig * neg_sig
    S = np.diag(diag)
    H = tx.T.dot(S.dot(tx))
    return H

def calculate_logistic_gradient_hessian(y, tx, w):
    """return the loss, gradient, and hessian."""
    loss, gradient = calculate_logistic_gradient(y, tx, w)
    return loss, gradient, calculate_hessian(tx, w)

def newton_method_iteration(y, tx, w):
    # return loss, gradient and hessian:
    loss, gradient, hessian = calculate_logistic_gradient_hessian(y, tx, w)

    # update w:
    a = hessian
    b = hessian.dot(w) - gradient

    w = np.linalg.solve(a, b)

    return loss, w

def learning_with_newton_method(y, tx, w, max_iter):

    losses = np.array([])
    threshold = 1e-08
    for iter in range(max_iter):
        loss, w = newton_method_iteration(y, tx, w)
        # log info
        if iter % 1 == 0:
            print("Current iteration={i}, the loss={l}".format(i=iter, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    return loss, w

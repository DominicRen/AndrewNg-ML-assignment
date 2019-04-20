import numpy as np
import train_linear_reg as tlr
import linear_reg_cost_function as lrcf


def learning_curve(X, y, Xval, yval, lmd):
    # Number of training examples
    m = X.shape[0]

    # You need to return these values correctly
    error_train = np.zeros(m)
    error_val = np.zeros(m)

    # ===================== Your Code Here =====================
    # Instructions : Fill in this function to return training errors in
    #                error_train and the cross validation errors in error_val.
    #                i.e., error_train[i] and error_val[i] should give you
    #                the errors obtained after training on i examples
    #
    # Note : You should evaluate the training error on the first i training
    #        examples (i.e. X[:i] and y[:i])
    #
    #        For the cross-validation error, you should instead evaluate on
    #        the _entire_ cross validation set (Xval and yval).
    #
    # Note : If you're using your cost function (linear_reg_cost_function)
    #        to compute the training and cross validation error, you should
    #        call the function with the lamdba argument set to 0.
    #        Do note that you will still need to use lamdba when running the
    #        training to obtain the theta parameters.
    #

    for i in range(m):
        x_i = X[:i+1]
        y_i = y[:i+1]
        theta = tlr.train_linear_reg(x_i, y_i, lmd)

        error_train[i] = lrcf.linear_reg_cost_function(theta, x_i, y_i, 0)[0]
        error_val[i] = lrcf.linear_reg_cost_function(theta, Xval, yval, 0)[0]

    # ==========================================================

    return error_train, error_val
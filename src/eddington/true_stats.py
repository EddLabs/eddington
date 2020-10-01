import numpy as np
from scipy.optimize import least_squares
from scipy.linalg import inv




def chi2(params, *args):
    #function, x, y, x_variance, y_variance, func_derivative_by_x
    func, x, y, xvar, yvar, derr = args

    return (y - func(params, x))/np.sqrt(xvar*derr(params, x) + yvar)




class ModifiedODR():
    def __init__(self, data, model, beta0=None):
        self.data = data
        self.model = model
        if beta0 is None: do_something
        self.beta0 = beta0

    def run(self):
        data  = self.data
        model = self.model
        
        #fit. FOR NOW, NUMERICALLY DERIVATES CHI2 (REQUIRES f_xa FOR ANALYTICAL)
        res = least_squares(chi2, self.beta0, '3-point', method='trf',
                            args=(model.fcn,
                                  data.x, data.y,
                                  np.square(data.sx), np.square(data.sy),
                                  model.jacd))

        #calculate approximate covariance matrix using Gauss-Newton approximation
        cov_mat = inv(res.jac.T@res.jac)

        #make "output" object
        output = object()
        output.beta = res.x
        output.chi2 = res.cost
        output.sd_beta = np.sqrt(np.diag(cov_mat)) #std(x) = sqrt(cov(x, x))
        output.cov_beta = cov_mat
        return output

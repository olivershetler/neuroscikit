import numpy as np

def mahal(Y, X):
    """MAHAL Mahalanobis distance
    D2 = MAHAL(Y,X) returns the Mahalanobis distance (in squared units) of
    each observation (point) in Y from the sample data in X, i.e.,

       D2(I) = (Y(I,:)-MU) * SIGMA^(-1) * (Y(I,:)-MU)',

    where MU and SIGMA are the sample mean and covariance of the data in X.
    Rows of Y and X correspond to observations, and columns to variables.  X
    and Y must have the same number of columns, but can have different numbers
    of rows.  X must have more rows than columns.

    Example:  Generate some highly correlated bivariate data in X.  The
    observations in Y with equal coordinate values are much closer to X as
    defined by Mahalanobis distance, compared to the observations with opposite
    coordinate values, even though they are all approximately equidistant from
    the mean using Euclidean distance.

       x = mvnrnd([0;0], [1 .9;.9 1], 100);
       y = [1 1;1 -1;-1 1;-1 -1];
       MahalDist = mahal(y,x)
       sqEuclidDist = sum((y - repmat(mean(x),4,1)).^2, 2)
       plot(x(:,1),x(:,2),'b.',y(:,1),y(:,2),'ro')


    Args:
        FD (ndarray): N by D array of feature vectors (N spikes, D dimensional feature space)
        ClusterSpikes (ndarray): Index into FD which lists spikes from the cell whose quality is to be evaluated.

    Returns:
        IsoDist: the isolation distance

    """
    rx, cx = X.shape
    ry, cy = Y.shape

    if cx != cy:
        raise ValueError('Mahal: Input Size Mismatch!')

    if rx < cx:
        raise ValueError('Mahal: Too few rows!')

    if len(np.where(np.iscomplex(X) == True)[0]) > 0:
        raise ValueError('Mahal: no complex values (X)!')
    elif len(np.where(np.iscomplex(Y) == True)[0]) > 0:
        raise ValueError('Mahal: no complex values (Y)!')

    m = np.mean(X, axis=0)

    M = np.tile(m, (ry, 1))
    C = X - np.tile(m, (rx, 1))

    Q, R = np.linalg.qr(C)

    ri = np.linalg.lstsq(R.T, (Y - M).T, rcond=None)[0]

    d = np.sum(np.multiply(ri, ri), axis=0).T * (rx - 1)

    return d

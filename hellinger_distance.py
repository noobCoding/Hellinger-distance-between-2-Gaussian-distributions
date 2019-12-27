import numpy as np

def hellinger_dist (X, Y):
    """ Calculates Hellinger distance between 2 multivariate normal distribution         
         X = X(x1, x2)
         Y = Y(y1, y2)         
         The definition can be found at https://en.wikipedia.org/wiki/Hellinger_distance
    """
    if len(X) < 2 or len(Y) < 2:      return 1.
    
    meanX = np.mean(X, axis=0)
    covX = np.cov(X, rowvar=0)
    detX = np.linalg.det(covX)
    
    meanY = np.mean(Y, axis=0)
    covY = np.cov(Y, rowvar=0)
    detY = np.linalg.det(covY)
    
    detXY = np.linalg.det((covX + covY)/2)
    if (np.linalg.det(covX + covY)/2) != 0:
            covXY_inverted = np.linalg.inv((covX + covY)/2)
    else:
            covXY_inverted = np.linalg.pinv((covX + covY)/2)    
    dist = 1. - (detX**.25 * detY**.25 / detXY**.5) * np.exp(-.125 * np.dot(np.dot(np.transpose(meanX-meanY),covXY_inverted),(meanX - meanY)))        
    return min(max(dist, 0.), 1.)
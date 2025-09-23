import numpy as np

def sample_Sz_weighted(sign, size=1):
    costheta = sign * np.random.triangular(0, 1, 1, size)
    theta = np.arccos(costheta)
    phi = np.random.uniform(0, 2 * np.pi, size)
    S = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), costheta]).T
    return S

def random_spherical(r, shape):
    out = np.random.normal(0, 1, shape)
    norm = np.linalg.norm(out, axis=-1)
    out *= r / norm[..., None]
    return out

def sample_xp(qbar, pbar, gamma, shape, beads=1):
    '''
        Samples one dimensional x (q) and p using Eq. (37) of 
        https://doi.org/10.1063/5.0139734 (A mapping approach to surface hopping)

        Used for Tully models.

        Arguments:
            qbar -- mean of q (position)
            pbar -- mean of p (momentum)
            gamma -- parameter defining the variance of q and p
            count -- number of samples

        Output:
            qs -- array of length count samples of q
            ps -- array of length count samples of p
    '''
    stdev_q = np.sqrt(beads / (2 * gamma))
    stdev_p = np.sqrt(beads * gamma / 2)
    return np.random.normal(qbar, stdev_q, shape), np.random.normal(pbar, stdev_p, shape)

"""Native fisheye bearings, including the calibrated region beyond 90 degrees."""
import numpy as np


def projection_jacobian(camera, xyz):
    """d(native pixel)/d(camera XYZ) for the same atan2 model as project()."""
    x, y, z = xyz.T
    r = np.hypot(x, y)
    safe_r = np.maximum(r, 1e-12)
    norm2 = r*r+z*z
    theta = np.arctan2(r, z)
    t2 = theta*theta
    a, b, c, d = camera.D
    rho = theta*(1+t2*(a+t2*(b+t2*(c+t2*d))))
    derivative = 1+t2*(3*a+t2*(5*b+t2*(7*c+t2*9*d)))
    scale = rho/safe_r
    dr = (derivative*z*safe_r/norm2-rho)/(safe_r*safe_r)
    dx, dy, dz = dr*x/safe_r, dr*y/safe_r, -derivative/norm2
    jac = np.stack([np.c_[scale+x*dx, x*dy, x*dz],
                    np.c_[y*dx, scale+y*dy, y*dz]], axis=1)
    axial = r < 1e-8
    if axial.any():
        jac[axial] = 0
        jac[axial, 0, 0] = 1/z[axial]
        jac[axial, 1, 1] = 1/z[axial]
    return np.einsum("ij,njk->nik", camera.K[:2, :2], jac)


def bearings(camera, pixels):
    pixels = np.asarray(pixels, float)
    y = (pixels[:, 1]-camera.K[1, 2])/camera.K[1, 1]
    x = (pixels[:, 0]-camera.K[0, 2]-camera.K[0, 1]*y)/camera.K[0, 0]
    rho = np.hypot(x, y)
    lo, hi = np.zeros_like(rho), np.full_like(rho, camera.theta_limit)
    a, b, c, d = camera.D
    for _ in range(30):
        theta = (lo+hi)/2
        t2 = theta*theta
        below = theta*(1+t2*(a+t2*(b+t2*(c+t2*d)))) < rho
        lo, hi = np.where(below, theta, lo), np.where(below, hi, theta)
    theta = (lo+hi)/2
    scale = np.sin(theta)/np.maximum(rho, 1e-12)
    return np.c_[x*scale, y*scale, np.cos(theta)]

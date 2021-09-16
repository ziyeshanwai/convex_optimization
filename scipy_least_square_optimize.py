import numpy as np
from scipy.spatial.transform import Rotation as R
import scipy.optimize

def hat(v):
    """
    vecotrized version of the hat function, creating for a vector its skew symmetric matrix.

    Args:
        v (np.array<float>(..., 3, 1)): The input vector.

    Returns:
        (np.array<float>(..., 3, 3)): The output skew symmetric matrix.

    """
    E1 = np.array([[0., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
    E2 = np.array([[0., 0., 1.], [0., 0., 0.], [-1., 0., 0.]])
    E3 = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 0.]])

    return v[..., 0:1, :] * E1 + v[..., 1:2, :] * E2 + v[..., 2:3, :] * E3


def exp(v, der=False):
    """
    Vectorized version of the exponential map.

    Args:
        v (np.array<float>(..., 3, 1)): The input axis-angle vector.
        der (bool, optional): Wether to output the derivative as well. Defaults to False.

    Returns:
        R (np.array<float>(..., 3, 3)): The corresponding rotation matrix.
        [dR (np.array<float>(3, ..., 3, 3)): The derivative of each rotation matrix.
                                            The matrix dR[i, ..., :, :] corresponds to
                                            the derivative d R[..., :, :] / d v[..., i, :],
                                            so the derivative of the rotation R gained
                                            through the axis-angle vector v with respect
                                            to v_i. Note that this is not a Jacobian of
                                            any form but a vectorized version of derivatives.]

    """
    n = np.linalg.norm(v, axis=-2, keepdims=True)
    H = hat(v)

    with np.errstate(all='ignore'):
        R = np.identity(3) + (np.sin(n) / n) * H + ((1 - np.cos(n)) / n ** 2) * (H @ H)
    R = np.where(n == 0, np.identity(3), R)

    if der:
        sh = (3,) + tuple(1 for _ in range(v.ndim - 2)) + (3, 1)
        dR = np.swapaxes(np.expand_dims(v, axis=0), 0, -2) * H
        dR = dR + hat(np.cross(v, ((np.identity(3) - R) @ np.identity(3).reshape(sh)), axis=-2))
        dR = dR @ R

        n = n ** 2  # redifinition
        with np.errstate(all='ignore'):
            dR = dR / n
        dR = np.where(n == 0, hat(np.identity(3).reshape(sh)), dR)

        return R, dR

    else:
        return R




# estimate v with least sqaures, so the objective function  becomes:
# minimize v over f(v) = sum_[1<=i<=n] (||p_1_i - exp(v)p_2_i||^2)
# Due to the way least_squres is implemented we have to pass the
# individual residuals ||p_1_i - exp(v)p_2_i||^2 as ||p_1_i - exp(v)p_2_i||.
from scipy.optimize import least_squares


def loss(x):
    r = exp(x.reshape(1, 3, 1))
    y = p_2 - r @ p_1
    y = np.linalg.norm(y, axis=-2).squeeze(-1)
    return y


def loss2(x):
    r = R.from_rotvec(x)
    ro = r.as_matrix()
    y = p_2 - ro @ p_1
    y = np.linalg.norm(y, axis=-2).squeeze(-1)

    return y


def loss2_hat(x):
    r = R.from_rotvec(x)
    ro = r.as_matrix()
    y = p_2 - ro @ p_1
    y = np.mean(np.linalg.norm(y, axis=-2).squeeze(-1))

    return y


def loss3(x):
    r = R.from_euler('zxy', x, degrees=True)
    ro = r.as_matrix()
    y = p_2 - ro @ p_1
    y = np.linalg.norm(y, axis=-2).squeeze(-1)

    return y


def loss4(x):
    r = x.reshape(3, 3)
    ro = r
    y = p_2 - ro @ p_1
    y = np.linalg.norm(y, axis=-2).squeeze(-1)

    return y


def loss5(x):

    r = R.from_quat(x)
    ro = r.as_matrix()
    y = p_2 - ro @ p_1
    y = np.linalg.norm(y, axis=-2).squeeze(-1)

    return y


def d_loss(x):
    R, d_R = exp(x.reshape(1, 3, 1), der=True)
    y = p_2 - R @ p_1
    d_y = -d_R @ p_1

    d_y = np.sum(y * d_y, axis=-2) / np.linalg.norm(y, axis=-2)
    d_y = d_y.squeeze(-1).T

    return d_y


def optimize_axis_angle_parameter():
    """
    using axis_angle parameter to optimize rotation matrix use custom exp
    :return:
    """
    x0 = np.zeros((3))
    res = least_squares(loss, x0, d_loss)  #  d_loss
    print("-" * 100)
    print("Axis-angle representation optimize rotation matrix custom exp")
    print('True axis-angle vector: {}'.format(v.reshape(-1)))
    print('Estimated axis-angle vector: {}'.format(res.x))
    print("-" * 100)
    print("\n")


def optimize_axis_angle_parameter_scipy():
    """
    using axis_angle parameter to optimize rotation matrix use custom exp
    :return:
    """
    x0 = np.zeros((3))
    res = least_squares(loss2, x0)  #  d_loss
    print("-" * 100)
    print("Axis-angle representation optimize rotation matrix use scipy R")
    print('True axis-angle vector: {}'.format(v.reshape(-1)))
    print('Estimated axis-angle vector: {}'.format(res.x))
    print("-" * 100)
    print("\n")


def minimize_axis_angle_parameter_scipy():
    """
    using axis_angle parameter to optimize rotation matrix use custom exp
    :return:
    """
    x0 = np.zeros((3))
    res = scipy.optimize.minimize(loss2_hat, x0, method="BFGS")  #  d_loss
    print("-" * 100)
    print("Axis-angle representation optimize rotation matrix use scipy R use scipy.optimize.minimize")
    print('True axis-angle vector: {}'.format(v.reshape(-1)))
    print('Estimated axis-angle vector: {}'.format(res.x))
    print("-" * 100)
    print("\n")


def optimize_yaw_pitch_roll_parameter_scipy():
    """
    using ayaw_pitch_roll parameter to optimize rotation matrix use custom exp
    :return:
    """
    x0 = np.zeros((3))
    res = least_squares(loss3, x0)  #  d_loss
    print("-" * 100)
    print("yaw_pitch_roll representation optimize rotation matrix use scipy R")
    print('True euler vector: {}'.format(euler_gt))
    print('Estimated uler vector : {}'.format(res.x))
    print('True axis-angle vector: {}'.format(v.reshape(-1)))
    r = R.from_euler('zxy', res.x, degrees=True)
    print('Estimated axis-angle vector : {}'.format(r.as_rotvec()))
    print("-" * 100)
    print("\n")


def optimize_matrix_parameter_scipy():
    """
    using rotation matrix parameter to optimize rotation matrix use custom exp
    :return:
    """
    x0 = np.zeros((9))
    res = least_squares(loss4, x0)  #  d_loss
    print("-" * 100)
    print("rotation matrix representation optimize rotation matrix use scipy R")
    print('True axis-angle vector: {}'.format(v.reshape(-1)))
    r = R.from_matrix(res.x.reshape(3, 3))
    print('Estimated axis-angle vector : {}'.format(r.as_rotvec()))
    print("-" * 100)
    print("\n")


def optimize_quaternions_parameter_scipy():
    """
    using Quaternions parameter to optimize rotation matrix use custom exp
    :return:
    """
    x0 = np.array([0, 0, 0, 1], dtype=np.float32)
    res = least_squares(loss5, x0)  #  d_loss
    print("-" * 100)
    print("Quaternions representation optimize rotation matrix use scipy R")
    print('True axis-angle vector: {}'.format(v.reshape(-1)))
    r = R.from_quat(res.x)
    print('Estimated axis-angle vector : {}'.format(r.as_rotvec()))
    print("-" * 100)
    print("\n")


if __name__ == "__main__":
    # generate two sets of points which differ by a rotation
    np.random.seed(1001)
    n = 100  # number of points
    p_1 = np.random.randn(n, 3, 1)
    v = np.array([0.9, 0.4, 0.008]).reshape(3, 1)  # the axis-angle vector
    r = R.from_rotvec(v[:, 0])
    euler_gt = r.as_euler("zxy", degrees=True)
    p_2 = exp(v) @ p_1 + np.random.randn(n, 3, 1) * 1e-5
    """
    least square
    """
    optimize_axis_angle_parameter()
    optimize_axis_angle_parameter_scipy()
    optimize_yaw_pitch_roll_parameter_scipy()
    optimize_matrix_parameter_scipy()
    optimize_quaternions_parameter_scipy()

    """
    use minimize
    """
    minimize_axis_angle_parameter_scipy()
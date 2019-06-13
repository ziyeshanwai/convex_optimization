import cvxpy as cp
from Util.util import *
import numpy as np

"""
使用偏移量模型
"""
if __name__ == "__main__":
    number_blendshapes = 14
    blendshape_path = "\\\\192.168.20.63\\ai\\Liyou_wang_data\\blendshapes"
    mesh_path = "\\\\192.168.20.63\\ai\\Liyou_wang_data\\warp_face_aligned"
    out_path = "./output"
    v0, f0 = loadObj(os.path.join(blendshape_path, "0.obj"))
    V = np.array(v0, dtype=np.float32).flatten()
    for i in range(1, number_blendshapes):
        v, f = loadObj(os.path.join(blendshape_path, "{}.obj".format(i)))
        v = np.array(v, dtype=np.float32).flatten()
        V = np.vstack((V, v))
    V = np.delete(V, 0, axis=0)
    b0 = np.array(v0, dtype=np.float32).flatten()
    delta_V = V - b0
    true_mesh, _ = loadObj(os.path.join(mesh_path, "align-1.obj"))
    true_mesh = np.array(true_mesh, dtype=np.float32).reshape(1, -1)
    coe = cp.Variable((1, number_blendshapes-1))
    cost = cp.sum_squares(coe * delta_V + b0[np.newaxis, :] - true_mesh)
    prob = cp.Problem(cp.Minimize(cost))
    prob.solve()
    # Print result.
    print("\nThe optimal value is", prob.value)
    print("The optimal x is")
    print(coe.value)
    print("status: {}".format(prob.status))
    # print("The norm of the residual is ", cp.norm(A * x - b, p=2).value)

    calculated_mesh = coe.value.dot(delta_V) + b0[np.newaxis, :]
    calculated_mesh = calculated_mesh.reshape(-1, 3)
    writeObj(os.path.join(out_path, "5.obj"), calculated_mesh, f0)




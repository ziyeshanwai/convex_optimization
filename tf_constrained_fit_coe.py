import tensorflow as tf
from Util.util import *
import numpy as np

"""
使用偏移量模型
参考链接：https://www.tensorflow.org/api_docs/python/tf/contrib/opt/ScipyOptimizerInterface
"""
if __name__ == "__main__":
    number_blendshapes = 39
    blendshape_path = "\\\\192.168.20.63\\ai\\Liyou_wang_data\\blendshapes"
    mesh_path = "\\\\192.168.20.63\\ai\\Liyou_wang_data\\origin_wrap_face_align"
    out_path = "\\\\192.168.20.63\\ai\Liyou_wang_data\\tf_output_v3"
    v0, f0 = loadObj(os.path.join(blendshape_path, "0.obj"))
    V = np.array(v0, dtype=np.float32).flatten()
    for i in range(1, number_blendshapes):
        v, f = loadObj(os.path.join(blendshape_path, "{}.obj".format(i)))
        v = np.array(v, dtype=np.float32).flatten()
        V = np.vstack((V, v))
    V = np.delete(V, 0, axis=0)
    b0 = np.array(v0, dtype=np.float32).flatten()
    delta_V = V - b0
    b0_tf = tf.constant(value=b0, name="mean_mesh")

    delta_v_tf = tf.constant(value=delta_V, name="blend_shapes")
    ini_coes = np.random.rand(1, number_blendshapes-1).astype(np.float32)
    coes = tf.get_variable(initializer=ini_coes, name="coes", trainable=True)
    pre_mesh = tf.matmul(coes, delta_v_tf) + b0_tf
    true_mesh = tf.placeholder(shape=[1, pre_mesh.shape[1]], dtype=tf.float32, name="input")
    # print(true_mesh)
    loss = tf.reduce_mean(tf.squared_difference(1000*pre_mesh, 1000*true_mesh))
    optimizer = tf.contrib.opt.ScipyOptimizerInterface(loss, var_to_bounds={coes: (0, 1)})
    tf_config = tf.ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        sess.run(tf.global_variables_initializer())
        names = names = ["{}.obj".format(n) for n in range(0, 1121)]
        COES = []
        for name in names:
            last_loss = 100
            count = 0
            tr_mesh, _ = loadObj(os.path.join(mesh_path, name))
            tr_mesh = np.array(tr_mesh, dtype=np.float32).reshape(1, -1)
            while True:
                optimizer.minimize(sess, feed_dict={true_mesh: tr_mesh})
                diff = sess.run(loss, feed_dict={true_mesh: tr_mesh})
                if np.abs(diff - last_loss) < 1e-8:
                    print("ite is over in {} times".format(count))
                    coe = sess.run(coes)
                    COES.append(coe)
                    print("coe is {}".format(coe))
                    calculated_mesh = coe.dot(delta_V) + b0
                    calculated_mesh = calculated_mesh.reshape(-1, 3)
                    writeObj(os.path.join(out_path, "tf-{}".format(name)), calculated_mesh, f0)
                    break
                else:
                    last_loss = diff
                    count += 1
                    # print("loss:{}".format(diff))
        save_pickle_file(os.path.join(out_path, "coe.pkl"), COES)




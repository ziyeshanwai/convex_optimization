import cv2
import torch
import sys
sys.path.append("./..")
sys.path.append(".")
from manopth.manolayer import ManoLayer
from manopth import demo
from Util.util import *
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from load_2d_keypoints import GetJsonCorList, get_Cordinate_2d


def LoadXML(file_name, node_name):
    """

    :param file_name: 读取的xml文件的路径和名称
    :param node_name: 读取的xml文件的节点名字
    :return: 返回对应节点名称的内容
    """
    # just like before we specify an enum flag, but this time it is
    # FILE_STORAGE_READ
    cv_file = cv2.FileStorage(file_name, cv2.FILE_STORAGE_READ)
    # for some reason __getattr__ doesn't work for FileStorage object in python
    # however in the C++ documentation, getNode, which is also available,
    # does the same thing
    # note we also have to specify the type to retrieve other wise we only get a
    # FileNode object back instead of a matrix
    matrix = cv_file.getNode(node_name).mat()
    cv_file.release()
    return matrix


def DumpXML(file_name, matrix, node_name):
    """

    :param file_name: 需要保存的文件名称
    :param matrix: 需要保存的矩阵
    :param node_name: 需要保存的节点名称
    :return: 无
    """
    # notice how its almost exactly the same, imagine cv2 is the namespace for cv
    # in C++, only difference is FILE_STORGE_WRITE is exposed directly in cv2
    cv_file = cv2.FileStorage(file_name, cv2.FILE_STORAGE_WRITE)
    # this corresponds to a key value pair, internally opencv takes your numpy
    # object and transforms it into a matrix just like you would do with <<
    # in c++
    cv_file.write(node_name, matrix)
    # note you *release* you don't close() a FileStorage object
    cv_file.release()


def get_Cordinate(Frames_keypoints, i):
    frame = Frames_keypoints[i]
    Cordinates = []
    confidences = []
    for j in range(0, len(frame), 4):
        Cordinates.append([frame[j], frame[j + 1], frame[j + 2]])
        confidences.append(frame[j+3])
    Cors = np.array(Cordinates, dtype=np.float32)
    confiden = np.array(confidences, dtype=np.float32)
    return Cors, confiden


def Project_to_2d(hand_cor, M):
    """

    :param cors: 根据2D点计算出来的3D手的坐标关键点
    :param hand_cor: Mano的手的关键点坐标
    :param M: 相机内外参矩阵
    :return: 返回uv Mano的手的uv 坐标
    """

    ones = np.ones((21, 1))
    hand_Homo_coor = np.hstack((hand_cor, ones))
    project_point = M.dot(hand_Homo_coor.T)
    uv_cor = (1 / project_point.T[:, 2][:, np.newaxis]) * project_point.T
    return uv_cor


def Project_to_2d_torch(hand_cor, camera_r_t, Intrinsics):
    """

    :param cors: 根据2D点计算出来的3D手的坐标关键点
    :param hand_cor: Mano的手的关键点坐标
    :param M: 相机内外参矩阵
    :return: 返回uv Mano的手的uv 坐标
    """
    print("camera.required_r_t:{}".format(camera_r_t.requires_grad))
    ones = torch.ones((21, 1))
    hand_Homo_coor = torch.cat((hand_cor, ones), dim=1)
    project_point = torch.from_numpy(Intrinsics.astype(np.float32)).mm(camera_r_t).mm(hand_Homo_coor.t())

    print("project_point:{}".format(torch.from_numpy(Intrinsics.astype(np.float32)).mm(camera_r_t).requires_grad))
    uv_cor = torch.mul(1 / project_point.t()[:, 2].unsqueeze(1), project_point.t())
    return uv_cor


if __name__ == "__main__":

    file_path = "D:\\pycharm_project\\Fit_hands\\manopth\\CameraMatrix"
    key_points_2d_json_file = os.path.join("../json_file", "2dkeypoints.json")
    file_name = "48072910100.xml"
    node_name_0 = "CameraMatrix"
    node_name_1 = "Intrinsics"
    node_name_2 = "Distortion"
    ncomps = 45
    mano_layer = ManoLayer(
        mano_root='D:\\pycharm_project\\Fit_hands\\manopth\\mano\\models', use_pca=True, ncomps=ncomps,
        flat_hand_mean=False)
    image = os.path.join("../image", "543.jpg")
    xml_file = os.path.join(file_path, file_name)
    came_mat = LoadXML(xml_file, node_name_0)
    Camera_r_t = torch.from_numpy(came_mat.astype(np.float32))
    Camera_r_t.requires_grad = True
    print(Camera_r_t)
    Intrinsics = LoadXML(xml_file, node_name_1)
    Distortion = LoadXML(xml_file, node_name_2)
    img = cv2.imread(image)
    undis_img = cv2.undistort(img, Intrinsics, Distortion)  # 去除畸变之后的图片数据
    M = Intrinsics.dot(came_mat)  # 投影矩阵 再乘以点的世界坐标
    M = M.astype(np.float32)
    print("load mano torch model... ")
    random_shape = torch.rand(1, 10)
    random_shape.requires_grad = True
    random_pose = torch.rand(1, ncomps + 3)
    random_pose.requires_grad = True
    hand_verts, hand_joints = mano_layer(random_pose, random_shape)
    face = mano_layer.th_faces + 1

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    loss_fn = torch.nn.MSELoss(reduce=False, size_average=True)
    learning_rate = 1e-4
    first_time = True
    i = 0
    Hand_joints_list = []
    # number_frames = len(Frames_keypoints)
    json_file = "..\\json_file\\3dkeypointsNot.json"
    cor = None
    f = open(json_file, 'r')
    Frames_keypoints = []
    for line in f.readlines():
        dic = json.loads(line)
        cor = dic['people'][0]["hand_right_keypoints_3d"]
        Frames_keypoints.append(cor)
    f.close()
    cors, _ = get_Cordinate(Frames_keypoints, 542)  # 抽取指定帧
    hand_verts, hand_joints = mano_layer(random_pose, random_shape)
    hand_cor = hand_joints[0, :, :]
    hand_cor_aligned, s, R, t = AlignTwoFaceWithFixedPoints(cors, hand_cor.detach().numpy(), [0, 2, 5, 9, 13, 17], non_linear_align=False,
                                                    return_sRt=True)
    s = s.astype(np.float32)
    R = R.astype(np.float32)
    t = t.astype(np.float32)
    # cors = s * R.dot(np.array(hand_cor.detach().numpy(), dtype=np.float32).T) + t  # 将mano手的世界坐标系转化为相机标定时用的世界坐标系
    # cors = cors.T  # 转置成为[-1, 3]的代码
    # uv_cor = Project_to_2d(cors, M)
    # for point in uv_cor:
    #     cv2.circle(undis_img, (int(point[0]), int(point[1])), radius=2, color=(0, 0, 255), thickness=5)
    # small_img = cv2.resize(undis_img, (0, 0), fx=0.5, fy=0.5)
    # cv2.imshow("dst", small_img)
    # cv2.waitKey(0)
    key_points_2d_list = GetJsonCorList(key_points_2d_json_file)
    number_frames = len(key_points_2d_json_file)
    for j in range(0, number_frames):
        Cors, confidences = get_Cordinate_2d(key_points_2d_list, 542)
        last_loss = 1000
        if i == 0:
            optimizer = torch.optim.Adam([random_shape, random_pose, Camera_r_t], lr=0.001, betas=(0.9, 0.999), eps=1e-08,
                                                     weight_decay=0, amsgrad=False)
            # optimizer = torch.optim.LBFGS([random_shape, random_pose, Camera_r_t], lr=0.001, max_iter=200)
            while True:
                optimizer.zero_grad()  # 梯度归0
                hand_verts, hand_joints = mano_layer(random_pose, random_shape)
                hand_cor = hand_joints[0, :, :]  # 21 * 3
                cors = s * torch.from_numpy(R).mm(hand_cor.t()) + torch.from_numpy(t)
                cors = cors.t()
                # cors = s * R.dot(np.array(hand_cor.detach().numpy(), dtype=np.float32).T) + t  # 将mano手的世界坐标系转化为相机标定时用的世界坐标系
                # cors = cors.T  # 转置成为[-1, 3]的代码

                # uv_cor = Project_to_2d_torch(cors, Camera_r_t, Intrinsics)  # 21 * 3

                ones = torch.ones((21, 1))
                hand_Homo_coor = torch.cat((cors, ones), dim=1)
                project_point = torch.from_numpy(Intrinsics.astype(np.float32)).mm(Camera_r_t).mm(hand_Homo_coor.t())
                uv_cor = torch.mul(1 / project_point.t()[:, 2].unsqueeze(1), project_point.t())
                # print(project_point.requires_grad)



                # cors = s * R.dot(hand_cor.T) + t
                loss = torch.mul(torch.sum(loss_fn(uv_cor[:, 0:2], torch.from_numpy(Cors)), dim=1), torch.from_numpy(confidences)).mean()
                if i % 100 == 0:
                    print("{} loss: {}".format(i, loss.item()))
                loss.backward()
                i = i + 1
                optimizer.step()  # 更新参数
                # print("Camera_r_t:{}".format(Camera_r_t))
                if np.abs(loss.detach().numpy() - last_loss) < learning_rate or i > 1000:
                    print("迭代满足要求，停止迭代")
                    for point in uv_cor:
                        cv2.circle(undis_img, (int(point[0]), int(point[1])), radius=2, color=(0, 0, 255), thickness=5)

                    for point in Cors:
                        cv2.circle(undis_img, (int(point[0]), int(point[1])), radius=2, color=(0, 255, 0), thickness=5)

                    small_img = cv2.resize(undis_img, (0, 0), fx=0.5, fy=0.5)
                    cv2.imshow("dst", small_img)
                    writeObj(os.path.join("../hand_output/h0.obj",), hand_verts.detach().numpy()[0, :, :].tolist(), face)
                    demo.display_hand({
                        'verts': hand_verts.detach().numpy(),
                        'joints': hand_joints.detach().numpy()
                    }, ax=ax, mano_faces=mano_layer.th_faces)
                    cv2.waitKey(0)
                    break
                else:
                    last_loss = loss.detach().numpy()

        else:
            if first_time:
                random_shape.requires_grad = False
                optimizer = torch.optim.Adam([random_pose], lr=0.001, betas=(0.9, 0.999), eps=1e-08,
                                            weight_decay=0, amsgrad=False)
                first_time = False
            i = 0
            while True:
                optimizer.zero_grad()  # 梯度归0
                random_pose.data[0][0:3] = 0
                hand_verts, hand_joints = mano_layer(random_pose, random_shape)
                h_joints = hand_joints.detach().numpy().reshape(-1, 3)
                cors = AlignTwoFaceWithFixedPoints(h_joints, Cors, [0, 2, 5, 9, 13, 17], non_linear_align=False)
                loss = torch.mul(torch.sum(loss_fn(hand_joints[0, :, :], torch.from_numpy(np.array(cors, dtype=np.float32))), dim=1), torch.from_numpy(confidences)).mean()
                if i % 100 == 0:
                    print("{} loss: {}".format(i, loss))
                loss.backward()
                i = i + 1
                optimizer.step()  # 更新参数

                if np.abs(loss.detach().numpy() - last_loss) < learning_rate or i > 700:
                    break
                else:
                    last_loss = loss.detach().numpy()
        Hand_joints_list.append(hand_joints.detach().numpy())

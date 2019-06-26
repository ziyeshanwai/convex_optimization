import torch
import sys
sys.path.append("./..")
from manopth.manolayer import ManoLayer
from manopth import demo
from Util.util import *
import json
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def animation_hands(Cors, ax, color):
    ax.scatter(Cors[:, 0], Cors[:, 1], Cors[:, 2], color='b')
    ax.plot(Cors[[0, 1, 2, 3, 4], 0], Cors[[0, 1, 2, 3, 4], 1], Cors[[0, 1, 2, 3, 4], 2], color=color)
    ax.plot(Cors[[0, 5, 6, 7, 8], 0], Cors[[0, 5, 6, 7, 8], 1], Cors[[0, 5, 6, 7, 8], 2], color=color)
    ax.plot(Cors[[0, 9, 10, 11, 12], 0], Cors[[0, 9, 10, 11, 12], 1], Cors[[0, 9, 10, 11, 12], 2], color=color)
    ax.plot(Cors[[0, 13, 14, 15, 16], 0], Cors[[0, 13, 14, 15, 16], 1], Cors[[0, 13, 14, 15, 16], 2], color=color)
    ax.plot(Cors[[0, 17, 18, 19, 20], 0], Cors[[0, 17, 18, 19, 20], 1], Cors[[0, 17, 18, 19, 20], 2], color=color)


def get_Cordinate(Frames_keypoints, i):
    frame = Frames_keypoints[i]
    Cordinates = []
    for j in range(0, len(frame), 4):
        Cordinates.append([frame[j], frame[j + 1], frame[j + 2]])
    Cors = np.array(Cordinates, dtype=np.float32)
    return Cors


if __name__ == "__main__":
    batch_size = 1
    # Select number of principal components for pose space
    ncomps = 45
    # Initialize MANO layer
    mano_layer = ManoLayer(
        mano_root='D:\\pycharm_project\\Fit_hands\\manopth\\mano\\models', use_pca=True, ncomps=ncomps,
        flat_hand_mean=False)
    json_file = "..\\json_file\\3dkeypoints.json"
    cor = None
    f = open(json_file, 'r')
    Frames_keypoints = []
    for line in f.readlines():
        dic = json.loads(line)
        cor = dic['people'][0]["hand_right_keypoints_3d"]
        Frames_keypoints.append(cor)
    f.close()

    random_shape = torch.rand(1, 10)
    random_shape.requires_grad = True
    # Generate random pose parameters, including 3 values for global axis-angle rotation
    random_pose = torch.rand(1, ncomps + 3)
    random_pose.requires_grad = True
    # Forward pass through MANO layer
    hand_verts, hand_joints = mano_layer(random_pose, random_shape)
    print(hand_joints.requires_grad)
    face = mano_layer.th_faces + 1

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    loss_fn = torch.nn.MSELoss(reduce=True, size_average=True)
    learning_rate = 1e-4
    first_time = True
    i = 0
    for j in range(0, len(Frames_keypoints)):
        Cors = get_Cordinate(Frames_keypoints, j)
        last_loss = 1000
        if i == 0:
            optimizer = torch.optim.Adam([random_shape, random_pose], lr=0.001, betas=(0.9, 0.999), eps=1e-08,
                                         weight_decay=0, amsgrad=False)
            while True:
                optimizer.zero_grad() # 梯度归0
                random_pose.data[0][0:3] = 0
                hand_verts, hand_joints = mano_layer(random_pose, random_shape)
                h_joints = hand_joints.detach().numpy().reshape(-1, 3)
                # h_joints = hand_joints.view(-1, 3)
                cors = AlignTwoFaceWithFixedPoints(h_joints, Cors, [0, 2, 5, 9, 13, 17], non_linear_align=False)
                # print(hand_joints.requires_grad)
                loss = loss_fn(hand_joints[0, :, :], torch.from_numpy(np.array(cors, dtype=np.float32)))
                if i % 100 == 0:
                    print("{} loss: {}".format(i, loss.item()))
                loss.backward()
                i = i + 1
                optimizer.step()  # 更新参数

                # print("grad is {}".format(random_pose.grad))
                # with torch.no_grad():
                #     random_shape -= learning_rate * random_shape.grad
                #     random_pose -= learning_rate * random_pose.grad
                #     # Manually zero the gradients after updating weights
                #     random_shape.grad.zero_()
                #     random_pose.grad.zero_()
                if np.abs(loss.item() - last_loss) < learning_rate or i > 700:
                    break
                else:
                    last_loss = loss.item()

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
                loss = loss_fn(hand_joints[0, :, :], torch.from_numpy(np.array(cors, dtype=np.float32)))
                if i % 100 == 0:
                    print("{} loss: {}".format(i, loss.item()))
                loss.backward()
                i = i + 1
                optimizer.step()  # 更新参数

                # print("grad is {}".format(random_pose.grad))
                # with torch.no_grad():
                #     random_shape -= learning_rate * random_shape.grad
                #     random_pose -= learning_rate * random_pose.grad
                #     # Manually zero the gradients after updating weights
                #     random_shape.grad.zero_()
                #     random_pose.grad.zero_()
                if np.abs(loss.item() - last_loss) < learning_rate or i > 700:
                    break
                else:
                    last_loss = loss.item()

        writeObj(os.path.join("D:\\pycharm_project\\Fit_hands\\manopth\\hand_output", "hand-{}.obj".format(j)), hand_verts.detach().numpy()[0, :, :].tolist(), face)
        print(hand_verts.detach().numpy()[0, :, :].shape)

    # animation_hands(h_joints, ax, color='r')
    animation_hands(np.array(cors), ax, color='y')
    # plt.show()

    demo.display_hand({
        'verts': hand_verts.detach().numpy(),
        'joints': hand_joints.detach().numpy()
    }, ax=ax, mano_faces=mano_layer.th_faces)


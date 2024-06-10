import torch
import torch.nn as nn

from core.config import cfg as cfg
from models import meshnet, posenet, gcndiff


class Diffpose2Mesh(nn.Module):
    def __init__(self, num_joint, graph_L, adj, config):
        super(Diffpose2Mesh, self).__init__()

        self.num_joint = num_joint
        self.diffpose = gcndiff(adj)
        self.pose2mesh = meshnet.get_model(num_joint_input_chan=2 + 3, num_mesh_output_chan=3, graph_L=graph_L)

    def forward(self, pose2d):
        pose3d = self.diffpose(pose2d)
        #pose3d = pose3d.reshape(-1, self.num_joint, 3)
        pose_combine = torch.cat((pose2d, pose3d.detach() / 1000), dim=2)
        cam_mesh = self.pose2mesh(pose_combine)

        return cam_mesh, pose3d


def get_model(num_joint, graph_L):
    model = Diffpose2Mesh(num_joint, graph_L)

    return model



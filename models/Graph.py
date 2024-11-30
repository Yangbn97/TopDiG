import torch
import torch.nn as nn
from copy import deepcopy
from pytorch_pretrained_vit import ViT
from torch.cuda.amp import autocast
from torchinfo import summary
from thop import profile,clever_format
from models.Sinkhorn import *
from models.ViT import transformer
from utils.setting_utils import *
from utils.poly_utils import *

class ScoreNet(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, 256, kernel_size=1, stride=1, padding=0, bias=True)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=True)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0, bias=True)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        n_points = x.shape[-1]

        x = x.unsqueeze(-1)
        x = x.repeat(1,1,1,n_points)
        t = torch.transpose(x, 2, 3)
        x = torch.cat((x, t), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        return x[:,0]


class Graph_Generator(nn.Module):

    def __init__(self,sinkhorn=True,featuremap_dim=64,configs=None):
        super(Graph_Generator, self).__init__()

        self.sinkhorn = sinkhorn
        self.configs = configs
        self.feature_dim = configs['Model']['hidden_dim']
        self.input_dim = featuremap_dim + 2
        self.sinkhorn_iterations = 100
        self.attention_layers = configs['Model']['num_attention_layers']
        self.num_heads = configs['Model']['num_heads']

        # Modules
        self.conv_init = nn.Sequential(
            nn.Linear(self.input_dim, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.feature_dim, self.feature_dim)
        )

        self.transformer = transformer(num_layers=self.attention_layers,num_heads=self.num_heads,dim=self.feature_dim)

        self.norm = nn.LayerNorm(self.feature_dim)

        self.conv_desc = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
            nn.ReLU(inplace=True)
        )

        self.scorenet1 = ScoreNet(self.feature_dim * 2)
        self.scorenet2 = ScoreNet(self.feature_dim * 2)
        self.weight_init()

    def weight_init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.normal_(m.bias, std=1e-6)
            # nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            

    def normalize_coordinates(self, graph, ws):
        graph = graph/ws
        return graph

    @autocast()
    def forward(self, image, descriptors, graph):
        B, _, H, W = image.shape
        B, N, _ = graph.shape

        if self.configs['Experiment']['object_type'] == 'line':
            sel_desc = getDescriptors_point(graph, descriptors)
        else:
            sel_desc = getVerticesDescriptors_grid(graph, descriptors)



        #  Multi-layer Transformer network.
        norm_graph = self.normalize_coordinates(graph, W)  # out: normalized coordinate system [0, 1]

        sel_desc = torch.cat(([sel_desc, norm_graph]),dim=2)

        sel_desc = self.conv_init(sel_desc)
        sel_desc = self.norm(self.transformer(sel_desc))
        sel_desc = self.conv_desc(sel_desc)


        sel_desc = sel_desc.permute(0, 2, 1)
        # Compute scores
        scores_1 = self.scorenet1(sel_desc)
        scores_2 = self.scorenet2(sel_desc)
        scores = scores_1 + torch.transpose(scores_2, 1, 2)


        if self.sinkhorn:
            P = log_optimal_transport(scores,100)
        else:
            P = scores

        return scores,P

if __name__ == "__main__":
    image = torch.randn(1,3,300,300)
    descriptors = torch.randn(1,64,300,300)
    vertices_pred = torch.randn(1,320,2)
    vertices_score = torch.randn(2,256,1)
    configs = {'Experiment':{'object_type':'water'},'Model':{'hidden_dim':768, 'num_attention_layers':4,'num_heads':12}}
    model = Graph_Generator(sinkhorn=False, configs=configs)
    # summary(model.transformer, input_size=(1,256,768))
    flops, params = profile(model, inputs=(image,descriptors,vertices_pred))
    macs, params_ = clever_format([flops, params], "%.3f")
    print('MACs:', macs)
    print('Paras:', params_)
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')
    # S,PM_pred = model(image,descriptors,vertices_pred)
    # print(S.shape,PM_pred.shape)
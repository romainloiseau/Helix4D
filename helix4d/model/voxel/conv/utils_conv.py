import spconv.pytorch as spconv
from torch import nn

class ConvLayer(nn.Module):

    def __init__(self, dim_in, dim_out, key=None, *args, **kwargs):
        super(ConvLayer, self).__init__()
        self.conv3x3x3 = conv3x3x3(dim_in, dim_out, indice_key=key+"_3x3x3")
        self.act = nn.LeakyReLU()
        self.bn = nn.BatchNorm1d(dim_out)

    def forward(self, x):
        x = self.conv3x3x3(x)
        x = x.replace_feature(self.bn(self.act(x.features)))
        return x

class AsymConvLayer(nn.Module):
    def __init__(self, dim_in, dim_out, key=None, *args, **kwargs):
        super(AsymConvLayer, self).__init__()

        self.conv_A0 = conv1x3x3(dim_in, dim_out, indice_key=key+"_1x3x3")
        self.bn_A0 = nn.BatchNorm1d(dim_out)
        self.act_A0 = nn.LeakyReLU()

        self.conv_A1 = conv3x1x3(dim_out, dim_out, indice_key=key+"_3x1x3")
        self.bn_A1 = nn.BatchNorm1d(dim_out)
        self.act_A1 = nn.LeakyReLU()

        self.conv_B0 = conv3x1x3(dim_in, dim_out, indice_key=key+"_3x1x3")
        self.bn_B0 = nn.BatchNorm1d(dim_out)
        self.act_B0 = nn.LeakyReLU()

        self.conv_B1 = conv1x3x3(dim_out, dim_out, indice_key=key+"_1x3x3")
        self.bn_B1 = nn.BatchNorm1d(dim_out)
        self.act_B1 = nn.LeakyReLU()

        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        A = self.conv_A0(x)
        A = A.replace_feature(self.bn_A0(self.act_A0(A.features)))

        A = self.conv_A1(A)
        A = A.replace_feature(self.bn_A1(self.act_A1(A.features)))

        B = self.conv_B0(x)
        B = B.replace_feature(self.bn_B0(self.act_B0(B.features)))

        B = self.conv_B1(A)
        B = B.replace_feature(self.bn_B1(self.act_B1(B.features)))

        A = A.replace_feature(A.features + B.features)
        return A

class MySubMConv3d(spconv.SubMConv3d):
    def __repr__(self):
        conv = "x".join([str(ks) for ks in self.kernel_size])
        return f"MySubMConv3d({self.in_channels}, {conv}, {self.out_channels})"

class MySparseConv3d(spconv.SparseConv3d):
    def __repr__(self):
        conv = "x".join([str(ks) for ks in self.kernel_size])
        return f"MySparseConv3d({self.in_channels}, {conv}, {self.out_channels})"

class MySparseInverseConv3d(spconv.SparseInverseConv3d):
    def __repr__(self):
        conv = "x".join([str(ks) for ks in self.kernel_size])
        return f"MySparseInverseConv3d({self.in_channels}, {conv}, {self.out_channels})"

def conv3x3x3(in_planes, out_planes, stride=1, indice_key=None):
    return MySubMConv3d(
        in_planes, out_planes, kernel_size=3, stride=stride,
        padding=1, bias=False, indice_key=indice_key
    )

def conv1x3x3(in_planes, out_planes, stride=1, indice_key=None):
    return MySubMConv3d(
        in_planes, out_planes, kernel_size=(1, 3, 3), stride=stride,
        padding=(0, 1, 1), bias=False, indice_key=indice_key
    )

def conv3x1x3(in_planes, out_planes, stride=1, indice_key=None):
    return MySubMConv3d(
        in_planes, out_planes, kernel_size=(3, 1, 3), stride=stride,
        padding=(1, 0, 1), bias=False, indice_key=indice_key
    )

def conv1x1x3(in_planes, out_planes, stride=1, indice_key=None):
    return MySubMConv3d(
        in_planes, out_planes, kernel_size=(1, 1, 3), stride=stride,
        padding=(0, 0, 1), bias=False, indice_key=indice_key
    )

def conv1x3x1(in_planes, out_planes, stride=1, indice_key=None):
    return MySubMConv3d(
        in_planes, out_planes, kernel_size=(1, 3, 1), stride=stride,
        padding=(0, 1, 0), bias=False, indice_key=indice_key
    )

def conv3x1x1(in_planes, out_planes, stride=1, indice_key=None):
    return MySubMConv3d(
        in_planes, out_planes, kernel_size=(3, 1, 1), stride=stride,
        padding=(1, 0, 0), bias=False, indice_key=indice_key
    )

class conv1x1x1(nn.Module):

    def __init__(self, in_planes, out_planes, *args, **kwargs):
        super().__init__()
        self.lin = nn.Linear(in_planes, out_planes)

    def forward(self, x):
        x = x.replace_feature(self.lin(x.features))
        return x

    def __repr__(self):
        return f"MySubMConv3d({self.lin.in_features}, 1x1x1, {self.lin.out_features})"
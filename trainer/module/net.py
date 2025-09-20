import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
import torchvision.models as models

def get_model(model_tag, num_classes):
    if model_tag == "ResNet20":
        model =  resnet20(num_classes)
    elif model_tag == "ResNet20_DIS":
        model =  resnet20_dis(num_classes)
    
    elif model_tag == "ResNet18":
        model =  resnet18(num_classes)
    elif model_tag == "ResNet18_DIS":
        model =  resnet18_dis(num_classes)

    elif model_tag == "ResNet50":
        model =  resnet50(num_classes)

    elif model_tag == "MLP":
        model =  MLP(num_classes=num_classes)
    elif model_tag == 'MLP_DIS':
        model =  MLP_DIS(num_classes = num_classes)
    
    elif model_tag == "CONV":
        model =  CONV(num_classes=num_classes)
    elif model_tag == 'CONV_DIS':
        model =  CONV_DIS(num_classes = num_classes)
    
    elif model_tag == "CONV2":
        model =  CONV(num_classes=num_classes)
    elif model_tag == 'CONV2_DIS':
        model =  CONV_DIS(num_classes = num_classes)
    
    else:
        raise NotImplementedError

    return model


class CONV_DIS(nn.Module):
    def __init__(self, num_classes = 10):
        super(CONV_DIS, self).__init__()
        self.conv1      = nn.Conv2d(3,8,4,1)
        self.bn1        = nn.BatchNorm2d(8)
        self.relu1      = nn.ReLU()
        self.dropout1   = nn.Dropout()
        self.avgpool1   = nn.AvgPool2d(2,2)
        self.conv2      = nn.Conv2d(8,32,4,1)
        self.bn2        = nn.BatchNorm2d(32)
        self.relu2      = nn.ReLU()
        self.dropout2   = nn.Dropout()
        self.avgpool2   = nn.AvgPool2d(2,2)
        self.conv3      = nn.Conv2d(32,64,4,1)
        self.relu3      = nn.ReLU()
        self.bn3        = nn.BatchNorm2d(64)
        self.dropout3   = nn.Dropout()
        self.avgpool3   = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(128, num_classes)


    def extract(self, x):
        x = self.conv1(x)     
        x = self.bn1(x)     
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.avgpool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)   
        x = self.dropout2(x)
        x = self.avgpool2(x)
        x = self.conv3(x)   
        x = self.relu3(x)   
        x = self.bn3(x)     
        x = self.dropout3(x)
        x = self.avgpool3(x)
        
        return x

    def predict(self, x):
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x

    def forward(self, x, mode=None, return_feat=False):
        feat = x = self.extract(x)
        x = x.view(x.size(0),-1)
        final_x = self.predict(x)
        if mode == 'tsne' or mode == 'mixup':
            return x, final_x
        else:
            if return_feat:
                return final_x, feat
            else:
                return final_x

class CONV2_DIS(nn.Module):
    def __init__(self, num_classes = 10):
        super(CONV2_DIS, self).__init__()
        self.conv1      = nn.Conv2d(3,8,7,1)
        self.bn1        = nn.BatchNorm2d(8)
        self.relu1      = nn.ReLU()
        self.dropout1   = nn.Dropout()
        self.avgpool1   = nn.AvgPool2d(3,3)
        self.conv2      = nn.Conv2d(8,32,7,1)
        self.bn2        = nn.BatchNorm2d(32)
        self.relu2      = nn.ReLU()
        self.dropout2   = nn.Dropout()
        self.avgpool2   = nn.AvgPool2d(3,3)
        self.conv3      = nn.Conv2d(32,64,5,1)
        self.relu3      = nn.ReLU()
        self.bn3        = nn.BatchNorm2d(64)
        self.dropout3   = nn.Dropout()
        self.conv4      = nn.Conv2d(64,128,3,1)
        self.relu4      = nn.ReLU()
        self.bn4        = nn.BatchNorm2d(128)
        self.dropout4   = nn.Dropout()
        self.avgpool   = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(256, num_classes)


    def extract(self, x):
        x = self.conv1(x)     
        x = self.bn1(x)     
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.avgpool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)   
        x = self.dropout2(x)
        x = self.avgpool2(x)
        x = self.conv3(x)   
        x = self.relu3(x)   
        x = self.bn3(x)     
        x = self.dropout3(x)
        x = self.conv4(x)   
        x = self.relu4(x)   
        x = self.bn4(x)     
        x = self.dropout4(x)
        x = self.avgpool(x)
        
        return x

    def predict(self, x):
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x

    def forward(self, x, mode=None, return_feat=False):
        feat = x = self.extract(x)
        x = x.view(x.size(0),-1)
        final_x = self.predict(x)
        if mode == 'tsne' or mode == 'mixup':
            return x, final_x
        else:
            if return_feat:
                return final_x, feat
            else:
                return final_x

class CONV2(nn.Module):
    def __init__(self, num_classes = 10):
        super(CONV2, self).__init__()
        self.conv1 = nn.Conv2d(3,8,7,1)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout()
        self.avgpool1 = nn.AvgPool2d(3,3)
        self.conv2 = nn.Conv2d(8,32,7,1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout()
        self.avgpool2 = nn.AvgPool2d(3,3)
        self.conv3 = nn.Conv2d(32,64,5,1)
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(64)
        self.dropout3 = nn.Dropout()
        self.conv4 = nn.Conv2d(64,128,3,1)
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(128)
        self.dropout4 = nn.Dropout()
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(128, num_classes)

    def extract(self, x):
        x = self.conv1(x)     
        x = self.bn1(x)     
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.avgpool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)   
        x = self.dropout2(x)
        x = self.avgpool2(x)
        x = self.conv3(x)   
        x = self.relu3(x)   
        x = self.bn3(x)     
        x = self.dropout3(x)
        x = self.conv4(x)   
        x = self.relu4(x)   
        x = self.bn4(x)     
        x = self.dropout4(x)
        x = self.avgpool(x)
        
        return x

    def predict(self, x):
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x

    def forward(self, x, mode=None, return_feat=False):
        feat = x = self.extract(x)
        x = x.view(x.size(0),-1)
        final_x = self.predict(x)
        if mode == 'tsne' or mode == 'mixup':
            return x, final_x
        else:
            if return_feat:
                return final_x, feat
            else:
                return final_x




class CONV(nn.Module):
    def __init__(self, num_classes = 10):
        super(CONV, self).__init__()
        self.conv1 = nn.Conv2d(3,8,4,1)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout()
        self.avgpool1 = nn.AvgPool2d(2,2)
        self.conv2 = nn.Conv2d(8,32,4,1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout()
        self.avgpool2 = nn.AvgPool2d(2,2)
        self.conv3 = nn.Conv2d(32,64,4,1)
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(64)
        self.dropout3 = nn.Dropout()
        self.avgpool3 = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(64, num_classes)

    def extract(self, x):
        x = self.conv1(x)     
        x = self.bn1(x)     
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.avgpool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)   
        x = self.dropout2(x)
        x = self.avgpool2(x)
        x = self.conv3(x)   
        x = self.relu3(x)   
        x = self.bn3(x)     
        x = self.dropout3(x)
        x = self.avgpool3(x)
        
        return x

    def predict(self, x):
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x

    def forward(self, x, mode=None, return_feat=False):
        feat = x = self.extract(x)
        x = x.view(x.size(0),-1)
        final_x = self.predict(x)
        if mode == 'tsne' or mode == 'mixup':
            return x, final_x
        else:
            if return_feat:
                return final_x, feat
            else:
                return final_x




class MLP_DIS(nn.Module):
    def __init__(self, num_classes = 10):
        super(MLP_DIS, self).__init__()
        self.Block = nn.Sequential(
            nn.Linear(3*28*28, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 16),
            nn.ReLU()
        )
        self.fc = nn.Linear(32, num_classes)

    def extract(self, x):
        x = x.view(x.size(0), -1)
        feat = self.Block(x)
        return feat

    def predict(self, x):
        prediction = self.fc(x)
        return prediction

    def forward(self, x, mode=None, return_feat=False):
        x = x.view(x.size(0), -1)
        feat = x = self.Block(x)
        final_x = self.fc(x)
        if mode == 'tsne' or mode == 'mixup':
            return x, final_x
        else:
            if return_feat:
                return final_x, feat
            else:
                return final_x

class MLP(nn.Module):
    def __init__(self, num_classes = 10):
        super(MLP, self).__init__()
        self.Block = nn.Sequential(
            nn.Linear(3*28*28, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 16),
            nn.ReLU()
        )
        self.fc = nn.Linear(16, num_classes)


    def extract(self, x):
        x = x.view(x.size(0), -1)
        feat = self.Block(x)
        return feat


    def predict(self, x):
        prediction = self.fc(x)
        return prediction
        
    def forward(self, x, mode=None, return_feat=False):
        x = x.view(x.size(0), -1)
        feat = x = self.Block(x)
        final_x = self.fc(x)
        if mode == 'tsne' or mode == 'mixup':
            return x, final_x
        else:
            if return_feat:
                return final_x, feat
            else:
                return final_x

def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option="A"):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == "A":
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(
                        x[:, :, ::2, ::2],
                        (0, 0, 0, 0, planes // 4, planes // 4),
                        "constant",
                        0,
                    )
                )
            elif option == "B":
                self.shortcut = nn.Sequential(
                    nn.Conv2d(
                        in_planes,
                        self.expansion * planes,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm2d(self.expansion * planes),
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out



def resnet18(num_classes):
    return ResNet18(num_classes)

def resnet18_dis(num_classes):
    return ResNet18_DIS(num_classes)


class ResNet18(nn.Module):
    def __init__(self,num_classes):
        super(ResNet18,self).__init__()
        self.fe = models.resnet18(pretrained=True)
        self.fe = nn.Sequential(*list(self.fe.children())[:-1])
        self.fc = nn.Linear(512, num_classes)


    def extract(self, x):
        out = self.fe(x)
        feat = out.view(out.size(0), -1)
        return feat

    def predict(self, x):
        prediction = self.fc(x)
        return prediction

    def forward(self, x, mode=None):
        out = self.fe(x)
        out = out.squeeze()
        final_out = self.fc(out)
        if mode == 'tsne' or mode == 'mixup':
            return out, final_out
        else:
            return final_out


class ResNet18_DIS(nn.Module):
    def __init__(self,num_classes):
        super(ResNet18_DIS,self).__init__()
        self.fe = models.resnet18(pretrained=True)
        self.fe = nn.Sequential(*list(self.fe.children())[:-1])
        self.fc = nn.Linear(1024, num_classes)


    def extract(self, x):
        out = self.fe(x)
        feat = out.view(out.size(0), -1)
        return feat

    def predict(self, x):
        prediction = self.fc(x)
        return prediction

    def forward(self, x, mode=None):
        out = self.fe(x)
        final_out = self.fc(out)
        if mode == 'tsne' or mode == 'mixup':
            return out, final_out
        else:
            return final_out


def resnet50(num_classes):
    return ResNet50(num_classes)


class ResNet50(nn.Module):
    def __init__(self,num_classes):
        super(ResNet50,self).__init__()
        self.fe = models.resnet50(pretrained=True)
        self.fe = nn.Sequential(*list(self.fe.children())[:-1])
        self.fc = nn.Linear(2048, num_classes)


    def extract(self, x):
        out = self.fe(x)
        feat = out.view(out.size(0), -1)
        return feat

    def predict(self, x):
        prediction = self.fc(x)
        return prediction

    def forward(self, x, mode=None):
        out = self.fe(x)
        out = out.squeeze()
        final_out = self.fc(out)
        if mode == 'tsne' or mode == 'mixup':
            return out, final_out
        else:
            return final_out



def resnet20(num_classes):
    return ResNet20(BasicBlock, [3, 3, 3], num_classes)

def resnet20_dis(num_classes):
    return ResNet20_DIS(BasicBlock, [3, 3, 3], num_classes)



class ResNet20_DIS(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet20_DIS, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(
            3, 16, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.avgpool =  nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(128, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def extract(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        feat = out.view(out.size(0), -1)

        return feat

    def predict(self, x):
        prediction = self.fc(x)
        return prediction

    def forward(self, x, mode=None):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        # out = F.avg_pool2d(out, out.size()[3])
        # out = out.view(out.size(0), -1)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        final_out = self.fc(out)
        if mode == 'tsne' or mode == 'mixup':
            return out, final_out
        else:
            return final_out


class ResNet20(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet20, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(
            3, 16, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.avgpool =  nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def extract(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        feat = out.view(out.size(0), -1)

        return feat

    def predict(self, x):
        prediction = self.fc(x)
        return prediction

    def forward(self, x, mode=None):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        final_out = self.fc(out)
        if mode == 'tsne' or mode == 'mixup':
            return out, final_out
        else:
            return final_out
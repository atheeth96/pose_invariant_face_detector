import torch
import torch.nn as nn


class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super().__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size
    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]
    
    
class  DetectionLoss(nn.Module):
    def __init__(self,model):
        super().__init__()
        
        loss_network = model
        for param in loss_network.parameters():
            param.requires_grad = False
        
        self.loss_network = loss_network.to('cuda')
        

    def forward(self,out_images, target_images):
        # Adversarial Loss
        output_features = self.loss_network(out_images)
        target_features = self.loss_network(target_images)
        loss=torch.nn.L1Loss().to('cuda')(output_features,target_features)
        
        return loss
    
    

    

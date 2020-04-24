import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

   
        
def save_model(model,optimizer,name,scheduler=None):
    if scheduler==None:
        checkpoint = {
        'state_dict': model.state_dict(),
        'optimizer' : optimizer.state_dict()}
    else:
        checkpoint = {
        'state_dict': model.state_dict(),
        'optimizer' : optimizer.state_dict(),
        'scheduler' : scheduler.state_dict()}

    torch.save(checkpoint,name)

def load_model(filename,model,optimizer=None,scheduler=None):
    checkpoint=torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])
    print("Done loading")
    if  optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(optimizer.state_dict()['param_groups'][-1]['lr'],' : Learning rate')
    if  scheduler:
        scheduler.load_state_dict(checkpoint['optimizer'])
        print(scheduler.state_dict()['param_groups'][-1]['lr'],' : Learning rate')
        
        
        
class Resnet34Triplet(nn.Module):
    """Constructs a ResNet-34 model for FaceNet training using triplet loss.
    Args:
        embedding_dimension (int): Required dimension of the resulting embedding layer that is outputted by the model.
                                   using triplet loss. Defaults to 128.
        pretrained (bool): If True, returns a model pre-trained on the ImageNet dataset from a PyTorch repository.
                           Defaults to False.
    """

    def __init__(self, embedding_dimension=128, pretrained=False):
        super().__init__()
        self.model = torchvision.models.resnet34(pretrained=pretrained)
        input_features_fc_layer = self.model.fc.in_features
        # Output embedding
        self.model.fc = nn.Linear(input_features_fc_layer, embedding_dimension)

    def l2_norm(self, input):
        """Perform l2 normalization operation on an input vector.
        code copied from liorshk's repository: https://github.com/liorshk/facenet_pytorch/blob/master/model.py
        """
        input_size = input.size()
        buffer = torch.pow(input, 2)
        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)
        _output = torch.div(input, norm.view(-1, 1).expand_as(input))
        output = _output.view(input_size)

        return output

    def forward(self, images):
        """Forward pass to output the embedding vector (feature vector) after l2-normalization and multiplication
        by scalar (alpha)."""
        embedding = self.model(images)
        embedding = self.l2_norm(embedding)
        # Multiply by alpha = 10 as suggested in https://arxiv.org/pdf/1703.09507.pdf
        #   Equation 9: number of classes in VGGFace2 dataset = 9131
        #   lower bound on alpha = 5, multiply alpha by 2; alpha = 10
        alpha = 10
        embedding = embedding * alpha

        return embedding
    
    

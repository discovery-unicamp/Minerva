from torch.nn.modules.loss import _Loss
import torch
from torch.nn import Parameter as torchnnParameter
from _topological_signature_distance import TopologicalSignatureDistance

class TopologicalLoss(_Loss):
    def __init__(self, p=2):
        super(TopologicalLoss, self).__init__()
        self.topological_signature_distance = TopologicalSignatureDistance()
        self.latent_norm = torch.nn.Parameter(data=torch.ones(1), requires_grad=True)


    def forward(self, x, x_encoded):
        x_distances = self._compute_distance_matrix(x)
        if len(x.size()) == 4:
            # If the input is an image (has 4 dimensions), normalize using theoretical maximum
            _, ch, b, w = x.size()
            # Compute the maximum distance we could get in the data space (this
            # is only valid for images wich are normalized between -1 and 1)
            max_distance = (2**2 * ch * b * w) ** 0.5
        else:
            # Else just take the max distance we got in the data
            max_distance = x_distances.max()
        x_distances = x_distances / max_distance
        # Latent distances
        x_encoded_distances = self._compute_distance_matrix(x_encoded)
        x_encoded_distances = x_encoded_distances / self.latent_norm 

        # Compute the topological signature distance
        topological_error, _ = self.topological_signature_distance(x, x_distances)
        # Normalize the topological error according to batch size
        topological_error = topological_error / x.size(0)
        return topological_error
    
    @staticmethod
    def _compute_distance_matrix(x, p=2):
        x_flat = x.view(x.size(0), -1)
        distances = torch.norm(x_flat[:, None] - x_flat, dim=2, p=p)
        return distances
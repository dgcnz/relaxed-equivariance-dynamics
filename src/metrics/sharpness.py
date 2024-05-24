import torch
import numpy as np

def get_sharpness(nn_model, datamodule, device):
    '''
    Calculates the sharpness of the loss landscape.
    
    Returns:
    float: The sharpness of the loss landscape as defined in the blog post.
    '''

    # List of magnitudes 
    T = torch.tensor(np.arange(0.1, 1.01, 0.3)).to(device)

    nn_model.eval()
    # Save the original state of the model
    original_state_dict = nn_model.state_dict()
    
    sharpness = 0
    num_samples = 0

    for (x, y) in datamodule.train_dataloader:
        sharpness_batch = 0
        num_samples += x.shape[0]

        x, y = x.to(device), y.to(device)

        # Calculate loss at w
        with torch.no_grad():
            predictions = nn_model(x)
            loss_w = torch.sqrt(torch.mean((predictions - y) ** 2))

            for param in nn_model.parameters():
                if param.requires_grad:
                    # Generate random unit vectors of the same shape as param
                    D = torch.randn(5, *param.shape).to(param.device)
                    # Unsqueeze norm multiple times so that division is done per entry for the first dim of D.
                    D = D / torch.norm(D.view(D.size(0), -1), dim=(1))[(..., ) + (None, ) * (len(D.size())-1)]
                    
                    for t in T:
                        for d in range(D.shape[0]):
                            # Apply the perturbation
                            param.add_(t*D[d])

                            predictions_perturbed = nn_model(x)
                            loss_perturbed = torch.sqrt(torch.mean((predictions_perturbed - y) ** 2))
                            sharpness_batch += abs(loss_perturbed - loss_w)
                            
                            nn_model.load_state_dict(original_state_dict)
                    
                    sharpness_batch /= D.shape[0] * len(T)

        sharpness += sharpness_batch

    sharpness /= num_samples
    
    return sharpness
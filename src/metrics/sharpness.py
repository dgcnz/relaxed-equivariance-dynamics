import torch
import numpy as np

def get_sharpness(nn_model, w, x, y):
    '''
    Calculates the sharpness of the loss landscape.
    
    Args:
    nn_model (torch.nn.Module): The neural network model.
    w (torch.Tensor): The weights of the model.
    x (torch.Tensor): The input data.
    y (torch.Tensor): The target data.
    
    Returns:
    float: The sharpness of the loss landscape as defined in the blog post.
    '''
    # List of magnitudes 
    T = np.arange(0.1, 1.01, 0.3) 

    w = torch.tensor(w, requires_grad=False).to(x.device)
    
    # Generate random unit vectors of the same shape as w
    D = torch.randn(5, *w.shape).to(w.device)        
    D = D / D.norm(dim=list(range(1, len(D.shape))), keepdim=True) 

    # Save the original state of the model
    original_state_dict = nn_model.state_dict()

    # Update the model's parameters with w
    nn_model.load_state_dict({'weight': w})
            
    # Calculate loss at w
    nn_model.eval()
    with torch.no_grad():
        predictions = nn_model(x)
        loss_w = torch.sqrt(torch.mean((predictions - y) ** 2))
    
    sharpness = 0
    for t in T:
        for d in D:
            w_plus_dt = w + t * d
            # Update the model's parameters with w_plus_dt
            nn_model.load_state_dict({'weight': w_plus_dt})     
            
            # Calculate loss at w + dt
            with torch.no_grad():
                predictions_plus_dt = nn_model(x)
                loss_plus_dt = torch.sqrt(torch.mean((predictions_plus_dt - y) ** 2))
            
            sharpness += abs(loss_plus_dt - loss_w)
    
    sharpness /= len(D) * len(T)
    
    # Restore the original model parameters
    nn_model.load_state_dict(original_state_dict)
    
    return sharpness.item()
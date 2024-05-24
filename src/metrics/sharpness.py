import torch
import numpy as np
import copy

def get_sharpness(model, datamodule, device):
    '''
    Calculates the sharpness of the loss landscape.
    
    Returns:
    float: The sharpness of the loss landscape as defined in the blog post.
    '''
    

    # List of magnitudes 
    T = torch.tensor(np.arange(0.1, 1.01, 0.3)).to(device)

    
    # Save the original state of the mode
    
    sharpness = 0
    num_samples = 0

    for (x ,y) in datamodule.train_dataloader():
        nn_model = copy.deepcopy(model)
        nn_model.eval()

        sharpness_batch = 0
        num_samples += x.shape[0]

        batch = (x.to(device), y.to(device))

        ### CODE IS WRONG BECAUSE WE SHOULD FIRST DO A FULL MODEL PERTUBATION, AND ONLY THEN DO THE BATCH SHARPENESS

        # Calculate loss at w
        with torch.no_grad():
            _, _, loss_w, _ = model.model_step(batch) #always the og model
            
            for t in T:
                for i in range(5):  #5 is the amount of directions we perturb in 
                    nn_model = perturb_model(model, t)
            
            _,_,loss_perturbed,_ = nn_model.model_step(batch)
            
            print('w', loss_w)
            print('perturbed', loss_perturbed)

        sharpness_batch += abs(loss_perturbed - loss_w)            
        sharpness_batch /= 5 * len(T)
        #print(sharpness_batch)
        sharpness += sharpness_batch

    sharpness /= num_samples
    
    return sharpness.cpu().detach().numpy()

def perturb_model(model, t):
    nn_model = copy.deepcopy(model)
    nn_model.eval()

    for param in nn_model.parameters():
        if param.requires_grad:
            # Generate random unit vectors of the same shape as param
            d = torch.randn(1, *param.shape).to(param.device) 
            # Unsqueeze norm multiple times so that division is done per entry for the first dim of D.
            d = d / torch.norm(d)
            d = d.squeeze(0)
            # Apply the perturbation
            param.add_(t*d)

    return nn_model







import numpy as np
from phi.flow import *
import torch
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def generate_translation_data(total_timesteps):
    try:
        os.mkdir("Translation")
    except:
        print("Translation folder already exists")


    inflow_lst = [(i, j) for i in np.arange(8,64,8) for j in np.arange(5,30,5)]

    res_x = 64
    res_y = 64
    pressure = None
    DOMAIN = dict(x=res_x, y=res_y)
    DOMAIN_LEFT = dict(x=res_x, y=res_y)
    DOMAIN_RIGHT = dict(x=res_x, y=res_y)

    LEFT = StaggeredGrid(HardGeometryMask(Box['x,y', :res_x//2, :]), extrapolation.ZERO, **DOMAIN_LEFT)
    RIGHT = StaggeredGrid(HardGeometryMask(Box['x,y', res_x//2:, :]), extrapolation.ZERO, **DOMAIN_RIGHT)

    smoke = CenteredGrid(0, extrapolation.BOUNDARY, x=res_x, y=res_y, bounds=Box['x,y', 0:res_x, 0:res_y])  # sampled at cell centers
    velocity = StaggeredGrid(0, extrapolation.ZERO, x=res_x, y=res_y, bounds=Box['x,y', 0:res_x, 0:res_y])# sampled in staggered form at face centers

    INFLOW_LOCATION = tensor(inflow_lst, batch('inflow_loc'), channel(vector='x,y'))
    #INFLOW_LOCATION.name = 'name'
    print(INFLOW_LOCATION)

    INFLOW = CenteredGrid(Sphere(center = INFLOW_LOCATION, radius=5), extrapolation.BOUNDARY, x=res_x, y=res_y, bounds=Box['x,y',0:res_x, 0:res_y])

    data = []
    for _ in view(smoke, velocity, description = 'hello very cool', gui='console').range(total_timesteps):
        smoke = advect.mac_cormack(smoke, velocity, dt=1) + INFLOW
        buoyancy_force1 = smoke * LEFT * (0, 0.1) @ velocity#(velocity * LEFT)
        buoyancy_force2 = smoke * RIGHT * (0, 0.2) @ velocity
        velocity = advect.semi_lagrangian(velocity, velocity, dt=0.5) + buoyancy_force1 + buoyancy_force2
        velocity, _ = fluid.make_incompressible(velocity, (), Solve('auto', 1e-5, 0, x0=pressure))
        data.append(torch.from_numpy(velocity.staggered_tensor().numpy(['inflow_loc', 'vector', 'y', 'x'])).float().unsqueeze(1))

    data = torch.cat(data, dim = 1)[:,:,:,:-1,:-1]
    for k in range(len(inflow_lst)):
        torch.save(data[k].double().float(), "Translation/raw_data_" + str(inflow_lst[k][0]) + "_" + str(inflow_lst[k][1]) + ".pt")

def generate_rotation_data(total_timesteps):
    try:
        os.mkdir("Rotation")
    except:
        print("Translation folder already exists")

    inflow_pos = [[(32,5), (0,0.01)], [(64-5,32), (-0.01, 0)], [(32,64-5),(0, -0.011)], [(5,32), (0.01, 0)]]
    #inflow pos is a task?
    #k is just a bunch of regeneration of the same shit? maybe since no seed its fine?

    k = 0
    for i in np.arange(9):
        for j in range(4):
            k += 1
            res_x = 64
            res_y = 64
            pressure = None
            DOMAIN = dict(x=res_x, y=res_y, bounds=Box['x,y',0:res_x, 0:res_y])
            smoke = CenteredGrid(0, extrapolation.BOUNDARY, x=res_x, y=res_y, bounds=Box['x,y', 0:res_x, 0:res_y])  # sampled at cell centers
            velocity = StaggeredGrid(0, extrapolation.ZERO, x=res_x, y=res_y, bounds=Box['x,y', 0:res_x, 0:res_y])  # sampled in staggered form at face centers
            INFLOW_LOCATION = tensor([inflow_pos[j][0]], batch('inflow_loc'), channel(vector='x,y'))
            INFLOW = CenteredGrid(Sphere(center=INFLOW_LOCATION, radius=5), extrapolation.BOUNDARY, x=res_x, y=res_y, bounds=Box['x,y',0:res_x, 0:res_y])

            data = []
            for _ in view(smoke, velocity, description = 'hello very cool', gui='console').range(total_timesteps):
                smoke = advect.mac_cormack(smoke, velocity, dt=1) + INFLOW
                buoyancy_force = smoke * tuple(e*k for e in inflow_pos[j][1]) @ velocity
                velocity = advect.semi_lagrangian(velocity, velocity, dt=1) + buoyancy_force
                velocity, _ = fluid.make_incompressible(velocity, (), Solve('auto', 1e-5, 0, x0=pressure))
                vel_tensor = torch.from_numpy(velocity.staggered_tensor().numpy(['inflow_loc', 'vector', 'y', 'x'])).float()
                data.append(vel_tensor)
            data = torch.cat(data, dim = 0)
            print(data.shape)
            print(data[1:,:,:64,:64].shape)
            torch.save(data.float()[1:,:,:64,:64], "Rotation/raw_data_"+ str(k) + "_" + str(j) + ".pt")

def generate_scale_data(total_timesteps):
    try:
        os.mkdir("Scale")
    except:
        print("Translation folder already exists")

    factors = np.linspace(1, 5, 41)
    for i in range(0, 38):
        factor = factors[i]
        res_x = int(64*factor)
        res_y = int(64*factor)
        pressure = None
        DOMAIN = dict(x=res_x, y=res_y, bounds=Box['x,y',0:res_x, 0:res_y])
        smoke = CenteredGrid(0, extrapolation.BOUNDARY, **DOMAIN)  # sampled at cell centers
        velocity = StaggeredGrid(0, extrapolation.ZERO, **DOMAIN)  # sampled in staggered form at face centers
        INFLOW_LOCATION = tensor([(32*factor, 5*factor)], batch('inflow_loc'), channel(vector='x,y'))
        INFLOW = CenteredGrid(Sphere(center=INFLOW_LOCATION, radius=5*factor), extrapolation.BOUNDARY, **DOMAIN)

        data = []
        for _ in view(smoke, velocity, description = 'hello very cool', gui='console').range(total_timesteps):
            smoke = advect.mac_cormack(smoke, velocity, dt=1) + INFLOW
            buoyancy_force = smoke * (0, 0.1*factor) @ velocity
            velocity = advect.semi_lagrangian(velocity, velocity, dt = 0.5) + buoyancy_force
            velocity, _ = fluid.make_incompressible(velocity, (), Solve('auto', 1e-5, 0, x0=pressure))
            vel_tensor = torch.from_numpy(velocity.staggered_tensor().numpy(['inflow_loc', 'vector', 'y', 'x'])).float()
            #print(vel_tensor.shape)
            data.append(vel_tensor)

        data = torch.cat(data, dim  = 0).transpose(0,1).unsqueeze(0)
        data = F.interpolate(data, scale_factor = (1, 1/factor, 1/factor), mode="trilinear", align_corners = True).squeeze(0).transpose(0,1)/factor
        #print(data.shape)
        torch.save(data.float()[1:,:,:64,:64], "Scale/raw_data_"+ str(i) + ".pt")

def generate_equivariance_test_data(total_timesteps):
    try:
        os.mkdir("equivariance_test")
    except:
        print("Translation folder already exists")

    for i in np.arange(50):
        res_x = 64
        res_y = 64
        pressure = None
        DOMAIN = dict(x=res_x, y=res_y, bounds=Box['x,y', 0:res_x, 0:res_y])
        smoke = CenteredGrid(0, extrapolation.BOUNDARY, x=res_x, y=res_y, bounds=Box['x,y', 0:res_x, 0:res_y])  # sampled at cell centers
        velocity = StaggeredGrid(0, extrapolation.ZERO, x=res_x, y=res_y, bounds=Box['x,y', 0:res_x, 0:res_y])  # sampled in staggered form at face centers
        INFLOW_LOCATION = tensor([(32, 5)], batch('inflow_loc'), channel(vector = 'x,y'))
        INFLOW = CenteredGrid(Sphere(center=INFLOW_LOCATION, radius=5), extrapolation.BOUNDARY, x=res_x, y=res_y, bounds=Box['x,y',0:res_x, 0:res_y])
        data = []
        for _ in view(smoke, velocity, description = 'hello very cool', gui='console').range(total_timesteps+10):
            smoke = advect.mac_cormack(smoke, velocity, dt=1) + INFLOW
            buoyancy_force = smoke * (0, 0.005*(i+1)) @ velocity
            velocity = advect.semi_lagrangian(velocity, velocity, dt=0.5) + buoyancy_force
            velocity, _ = fluid.make_incompressible(velocity, (), Solve('auto', 1e-5, 0, x0=pressure))
            vel_tensor = torch.from_numpy(velocity.staggered_tensor().numpy(['inflow_loc', 'vector', 'y', 'x'])).float()

            data.append(vel_tensor)
        data = torch.cat(data, dim = 0)
        torch.save(data.float()[10:total_timesteps+10,:,:64,:64], "equivariance_test/raw_data_"+ str(i) + ".pt")

        def rot_vector(inp, theta):
            #inp shape: c x 2 x 64 x 64
            theta = torch.tensor(theta).float()
            rot_matrix = torch.tensor([[torch.cos(theta), -torch.sin(theta)], [torch.sin(theta), torch.cos(theta)]]).float()
            out = torch.einsum("ab, bc... -> ac...",(rot_matrix, inp.transpose(0,1))).transpose(0,1)
            return out

        def get_rot_mat(theta):
            theta = torch.tensor(theta).float()
            return torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                                [torch.sin(theta), torch.cos(theta), 0]]).float()

        def rot_img(x, theta):
            rot_mat = get_rot_mat(theta)[None, ...].float().repeat(x.shape[0],1,1)
            grid = F.affine_grid(rot_mat, x.size()).float()
            x = F.grid_sample(x, grid)
            return x.float()

        def rot_field(x, theta):
            x_rot = torch.cat([rot_img(rot_vector(x, theta)[:,:1],  theta),
                            rot_img(rot_vector(x, theta)[:,-1:], theta)], dim = 1)
            return x_rot

        direc = "equivariance_test/raw_data_"
        lst = [(1,1,1,1), (1,2,3,4), (1,3,5,7), (1,4,7,10), (1,5,9,13), (1,6,11,16),
            (1,7,13,19), (1,8,15,22), (1,9,17,25), (1,10,19,28)]

        EE = []
        for i, item in enumerate(lst):
            d1 = torch.load(direc + str(item[0]) + ".pt")
            d2 = torch.load(direc + str(item[1]) + ".pt")
            d3 = torch.load(direc + str(item[2]) + ".pt")
            d4 = torch.load(direc + str(item[3]) + ".pt")
            torch.save(d1, "equivariance_test/E_" + str(i) + "/data_0.pt")
            torch.save(rot_field(d2, np.pi/2), "equivariance_test/E_" + str(i) + "/data_1.pt")
            torch.save(rot_field(d3, np.pi), "equivariance_test/E_" + str(i) + "/data_2.pt")
            torch.save(rot_field(d4, np.pi/2*3), "equivariance_test/E_" + str(i) + "/data_3.pt")
            # Equivariance error of each dataset
            EE.append(np.round(torch.mean((torch.abs(d1[10:-1]-d2[10:-1]) + torch.abs(d1[10:-1]-d3[10:-1]) + torch.abs(d1[10:-1]-d4[10:-1]))/3).numpy().item(),3))

import torch.utils.data as data
import torch
import h5py
import numpy as np
import random


class TrainDataFromHdf5(data.Dataset):
    def __init__(self, file_path, scale, patch_size, an):
        super(TrainDataFromHdf5, self).__init__()
        
        hf = h5py.File(file_path)
        self.img_HR = hf.get('img_HR')           # [N,ah,aw,h,w]
        self.img_LR = hf.get('img_LR_{}'.format(scale))   # [N,ah,aw,h/s,w/s]
        self.img_size = hf.get('img_size')  # [N,2]
        
        self.scale = scale        
        self.psize = patch_size
        self.an = an
    
    def __getitem__(self, index):
                        
        # get one item
        hr = self.img_HR[index]       # [ah,aw,h,w]
        lr = self.img_LR[index]   # [ah,aw,h/s,w/s]
                                               
        # crop to patch
        H, W = self.img_size[index]

        x = random.randrange(0, H-self.psize, 8)    
        y = random.randrange(0, W-self.psize, 8) 
        hr = hr[:, :, x:x+self.psize, y:y+self.psize] # [ah,aw,ph,pw]
        lr = lr[:, :, x//self.scale:x//self.scale+self.psize//self.scale, y//self.scale:y//self.scale+self.psize//self.scale] # [ah,aw,ph/s,pw/s]    

        # 4D augmentation
        # flip
        if np.random.rand(1)>0.5:
            hr = np.flip(np.flip(hr,0),2)
            lr = np.flip(np.flip(lr,0),2)
               
        if np.random.rand(1)>0.5:
            hr = np.flip(np.flip(hr,1),3)
            lr = np.flip(np.flip(lr,1),3)

        # rotate
        r_ang = np.random.randint(1,5)
        hr = np.rot90(hr,r_ang,(2,3))
        hr = np.rot90(hr,r_ang,(0,1))
        lr = np.rot90(lr,r_ang,(2,3))
        lr = np.rot90(lr,r_ang,(0,1))           
            
        # to tensor     
        hr = hr.reshape(-1,self.psize,self.psize) # [an,ph,pw]
        lr = lr.reshape(-1,self.psize//self.scale,self.psize//self.scale) #[an,phs,pws]

        ref_ind = np.random.randint(self.an*self.an)
        # ref_ind = int(24)
        hr = hr[ref_ind:ref_ind+1] #[1,ph,pw]
           
        hr = torch.from_numpy(hr.astype(np.float32)/255.0)
        lr = torch.from_numpy(lr.astype(np.float32)/255.0)  
        
        return hr, lr, ref_ind

    def __len__(self):
        return self.img_HR.shape[0]


class TrainDataFromHdf5_LF(data.Dataset):
    def __init__(self, file_path, scale, patch_size, an):
        super(TrainDataFromHdf5_LF, self).__init__()

        hf = h5py.File(file_path)
        self.img_HR = hf.get('img_HR')  # [N,ah,aw,h,w]
        self.img_LR = hf.get('img_LR_{}'.format(scale))  # [N,ah,aw,h/s,w/s]
        self.img_size = hf.get('img_size')  # [N,2]

        self.scale = scale
        self.psize = patch_size
        self.an = an

    def __getitem__(self, index):

        # get one item
        hr = self.img_HR[index]  # [ah,aw,h,w]
        lr = self.img_LR[index]  # [ah,aw,h/s,w/s]

        # crop to patch
        H, W = self.img_size[index]

        x = random.randrange(0, H - self.psize, 8)
        y = random.randrange(0, W - self.psize, 8)
        hr = hr[:, :, x:x + self.psize, y:y + self.psize]  # [ah,aw,ph,pw]
        lr = lr[:, :, x // self.scale:x // self.scale + self.psize // self.scale,
             y // self.scale:y // self.scale + self.psize // self.scale]  # [ah,aw,ph/s,pw/s]

        # 4D augmentation
        # flip
        if np.random.rand(1) > 0.5:
            hr = np.flip(np.flip(hr, 0), 2)
            lr = np.flip(np.flip(lr, 0), 2)

        if np.random.rand(1) > 0.5:
            hr = np.flip(np.flip(hr, 1), 3)
            lr = np.flip(np.flip(lr, 1), 3)

        # rotate
        r_ang = np.random.randint(1, 5)
        hr = np.rot90(hr, r_ang, (2, 3))
        hr = np.rot90(hr, r_ang, (0, 1))
        lr = np.rot90(lr, r_ang, (2, 3))
        lr = np.rot90(lr, r_ang, (0, 1))

        # to tensor
        hr = hr.reshape(-1, self.psize, self.psize)  # [an,ph,pw]
        lr = lr.reshape(-1, self.psize // self.scale, self.psize // self.scale)  # [an,phs,pws]

        hr = torch.from_numpy(hr.astype(np.float32) / 255.0)
        lr = torch.from_numpy(lr.astype(np.float32) / 255.0)

        return hr, lr

    def __len__(self):
        return self.img_HR.shape[0]

class TestDataFromHdf5(data.Dataset):
    def __init__(self, file_path, scale):
        super(TestDataFromHdf5, self).__init__()
        hf = h5py.File(file_path)

        self.GT_y = hf.get('/GT_y')  # [N,aw,ah,h,w]
        self.LR_ycbcr = hf.get('/LR_ycbcr')  # [N,ah,aw,3,h/s,w/s]

        self.scale = scale

    def __getitem__(self, index):
        h = self.GT_y.shape[3]
        w = self.GT_y.shape[4]

        gt_y = self.GT_y[index]
        gt_y = gt_y.reshape(-1, h, w)
        gt_y = torch.from_numpy(gt_y.astype(np.float32) / 255.0)

        lr_ycbcr = self.LR_ycbcr[index]
        lr_ycbcr = torch.from_numpy(lr_ycbcr.astype(np.float32) / 255.0)

        lr_y = lr_ycbcr[:, :, 0, :, :].clone().view(-1, h // self.scale, w // self.scale)

        lr_ycbcr_up = lr_ycbcr.view(1, -1, h // self.scale, w // self.scale)
        lr_ycbcr_up = torch.nn.functional.interpolate(lr_ycbcr_up, scale_factor=self.scale, mode='bicubic',
                                                      align_corners=False)
        lr_ycbcr_up = lr_ycbcr_up.view(-1, 3, h, w)

        return gt_y, lr_ycbcr_up, lr_y

    def __len__(self):
        return self.GT_y.shape[0]
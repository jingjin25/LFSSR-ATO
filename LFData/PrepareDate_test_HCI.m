%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% generate test data from HCI dataset
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ===> test_DatasetName.h5  
% uint8 0-255
%  ['GT_Y']     [w,h,aw,ah,N]
%  ['LR_ycbcr'] [w/scale,h/scale,3,aw,ah,N]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear;close all;

%% params
data_folder = 'dataset_test_hci';
scale = 2;
savepath = sprintf('test_HCI_x%d.h5',scale);
an = 7;

%% initilization
GT_y   = [];
LR_ycbcr = [];
count = 0;

data_list = dir(data_folder);
data_list = data_list(3:end);
%% generate data
for k = 1:length(data_list)
    lfname = data_list(k).name;
    read_path = fullfile(data_folder,lfname);    
    lf_gt_rgb = read_hci(read_path,9,an);
    lf_gt_ycbcr = rgb2ycbcr_5d(lf_gt_rgb);
    
    lf_gt_y = squeeze(lf_gt_ycbcr(:,:,1,:,:)); %[h,w,ah,aw]
    lf_lr_ycbcr = imresize(lf_gt_ycbcr,1/scale,'bicubic');  %[h/s,w/s,3,ah,aw]
    
    GT_y = cat(5,GT_y,lf_gt_y);
    LR_ycbcr = cat(6,LR_ycbcr,lf_lr_ycbcr);
end

GT_y = permute(GT_y,[2,1,4,3,5]); %[h,w,ah,aw,n]--->[w,h,aw,ah,n]
LR_ycbcr = permute(LR_ycbcr,[2,1,3,5,4,6]); %[h,w,3,ah,aw,n]--->[w,h,3,aw,ah,n]

%% save data
if exist(savepath,'file')
  fprintf('Warning: replacing existing file %s \n', savepath);
  delete(savepath);
end 
h5create(savepath,'/GT_y',size(GT_y),'Datatype','uint8');
h5create(savepath,'/LR_ycbcr',size(LR_ycbcr),'Datatype','uint8');

h5write(savepath, '/GT_y', GT_y);
h5write(savepath, '/LR_ycbcr', LR_ycbcr);

h5disp(savepath);

%% functions

function lf = read_hci(read_path, an_org, an_new)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% read [h,w,3,ah,aw] data from HCI data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    H = 512;
    W = 512;
    lf = zeros(H,W,3,an_org,an_org,'uint8');
    for v = 1:an_org
        for u = 1:an_org
            ind = (v-1)*an_org+(u-1);
            imgname = strcat('input_Cam',num2str(ind,'%03d'),'.png');
            sub = imread(fullfile(read_path,imgname));
            lf(:,:,:,v,u) = sub;
        end
    end
    an_crop = ceil((an_org - an_new) / 2 );
    lf = lf(:,:,:,1+an_crop:an_new+an_crop,1+an_crop:an_new+an_crop);
end

function lf_ycbcr = rgb2ycbcr_5d(lf_rgb)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% lf_rgb [h,w,3,ah,aw] --> lf_ycbcr [h,w,3,ah,aw]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if length(size(lf_rgb))<5
        error('input must have 5 dimensions');
    else
        lf_ycbcr = zeros(size(lf_rgb),'like',lf_rgb);
        for v = 1:size(lf_rgb,4)
            for u = 1:size(lf_rgb,5)
                lf_ycbcr(:,:,:,v,u) = rgb2ycbcr(lf_rgb(:,:,:,v,u));
            end
        end
    end
end
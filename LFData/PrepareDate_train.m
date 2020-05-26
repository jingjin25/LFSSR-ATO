%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% generate training data 
% containing 'Stanford', 'Kalantari', 'HCI'
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% output: train.h5 
% uint8 0-255
%  ['data_HR']   [w,h,aw,ah,N] 
%  ['data_LR_2'] [w/2,h/2,aw,ah,N] 
%  ['data_LR_4'] [w/4,h/4,aw,ah,N]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; close all;

%% path
dataset_list = {'Stanford', 'Kalantari','HCI'};
folder_list = {...
    './dataset_train/stanford',...
    './dataset_train/kalantari',...
    './dataset_train/hci'};
      
savepath = 'train_all.h5';
an = 7;  % angular number

%% initilization
data_HR = zeros(600,600,an,an,1,'uint8');
data_LR_2 = zeros(300,300,an,an,1,'uint8');
data_LR_4 = zeros(150,150,an,an,1,'uint8');
data_size = zeros(2,1,'uint16');
count = 0;

%% read datasets
for i_set = 1:length(dataset_list)
    dataset = dataset_list{i_set};
    folder = folder_list{i_set};
    
    %%% read list
    listname = ['dataset_train/trainList_',dataset,'.txt'];
    f = fopen(listname);
    C = textscan(f, '%s', 'CommentStyle', '#');
    list = C{1};
    fclose(f);
    
    %%% read lfs
    for i_lf = 1:length(list)
        lfname = list{i_lf};
        
        if strcmp(dataset,'Stanford') || strcmp(dataset,'Kalantari')
            read_path = sprintf('%s/%s.png',folder,lfname);
            lf_rgb = read_eslf(read_path,14,an);
        end
        
        if strcmp(dataset,'HCI')
            read_path = fullfile(folder,lfname);
            lf_rgb = read_hci(read_path,9,an);
        end
        
        lf_ycbcr = rgb2ycbcr_5d(lf_rgb);
        
        hr = squeeze(lf_ycbcr(:,:,1,:,:));
        lr_2 = imresize(hr,1/2,'bicubic');
        lr_4 = imresize(hr,1/4,'bicubic');
        H = size(hr,1);
        W = size(hr,2);
        
        count = count +1;
        data_HR(1:H,1:W,:,:,count) = hr;
        data_LR_2(1:H/2,1:W/2,:,:,count) = lr_2;
        data_LR_4(1:H/4,1:W/4,:,:,count) = lr_4;
        data_size(:,count)=[H,W];         
    end
    
end

%% generate data
order = randperm(count);
data_HR = permute(data_HR(:, :, :, :, order),[2,1,4,3,5]); %[h,w,ah,aw,N] -> [w,h,aw,ah,N]  
data_LR_2 = permute(data_LR_2(:, :, :, :, order),[2,1,4,3,5]);
data_LR_4 = permute(data_LR_4(:, :, :, :, order),[2,1,4,3,5]);
data_size = data_size(:,order);  %[2,N]

%% writing to HDF5
if exist(savepath,'file')
  fprintf('Warning: replacing existing file %s \n', savepath);
  delete(savepath);
end 

h5create(savepath, '/img_HR', size(data_HR), 'Datatype', 'uint8'); % width, height, channels, number 
h5create(savepath, '/img_LR_2', size(data_LR_2), 'Datatype', 'uint8'); % width, height, channels, number 
h5create(savepath, '/img_LR_4', size(data_LR_4), 'Datatype', 'uint8');    
h5create(savepath, '/img_size', size(data_size), 'Datatype', 'uint16');   

h5write(savepath, '/img_HR', data_HR);
h5write(savepath, '/img_LR_2', data_LR_2);  
h5write(savepath, '/img_LR_4', data_LR_4);
h5write(savepath, '/img_size', data_size);

h5disp(savepath);


%% functions

function lf = read_eslf(read_path, an_org, an_new)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% read [h,w,3,ah,aw] data from eslf data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    eslf = im2uint8(imread(read_path));

    H = size(eslf,1) / an_org;
    H = floor(H/4)*4;
    W = size(eslf,2) / an_org;
    W = floor(W/4)*4;

    lf = zeros(H,W,3,an_org,an_org,'uint8');
    for v = 1:an_org
        for u = 1:an_org
            sub = eslf(v:an_org:end, u:an_org:end, :);
            lf(:,:,:,v,u) = sub(1:H,1:W,:);
        end
    end
    an_crop = ceil((an_org - an_new) / 2 );
    lf = lf(:,:,:,1+an_crop:an_new+an_crop,1+an_crop:an_new+an_crop);

end


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


# rhis is the main py for trian pair to path
from model import cGAN_build2 # the mmodel

import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import Net_pair_path_body
import arg_parse
import imagenet
from analy import MY_ANALYSIS
from analy import Save_signal_enum
import cv2
import numpy
from image_trans import BaseTransform  
import os
from dataset_pair_path import myDataloader_for_shift,Batch_size,Resample_size, Path_length, Mat_size,Resample_size2
from dataset_pair_path import Original_window_Len
from multiscaleloss import multiscaleEPE
import torch.nn.functional as F
Continue_flag = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Switch control for the Visdom or Not
Visdom_flag  = True 

if Visdom_flag == True:
    from analy_visdom import VisdomLinePlotter
    plotter = VisdomLinePlotter(env_name='path finding training Plots')



from scipy import signal 
pth_save_dir = "../out/Deep_pair_to_path/"

if not os.path.exists(pth_save_dir):
    os.makedirs(pth_save_dir)
Matrix_dir =  "..\\dataset\\CostMatrix\\1\\"
#Save_pic_dir = '..\\DeepPathFinding_Version2\\out\\'
opt = arg_parse.opt
opt.cuda = True
# check the cuda device 
print(torch.cuda.current_device())
print(torch.cuda.device(0))
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
print(torch.cuda.is_available())
dataroot = "..\\dataset\\CostMatrix\\"

torch.set_num_threads(2)
######################################################################
# Data
# ----
# 
# In this tutorial we will use the `Celeb-A Faces
# dataset <http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html>`__ which can
# be downloaded at the linked site, or in `Google
# Drive <https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg>`__.
# The dataset will download as a file named *img_align_celeba.zip*. Once
# downloaded, create a directory named *celeba* and extract the zip file
# into that directory. Then, set the *dataroot* input for this notebook to
# the *celeba* directory you just created. The resulting directory
# structure should be:
# 
# ::
# 
#    /path/to/celeba
#        -> img_align_celeba  
#            -> 188242.jpg
#            -> 173822.jpg
#            -> 284702.jpg
#            -> 537394.jpg
#               ...
# 
# This is an important step because we will be using the ImageFolder
# dataset class, which requires there to be subdirectories in the
# dataset’s root folder. Now, we can create the dataset, create the
# dataloader, set the device to run on, and finally visualize some of the
# training data.
# 
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.002)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# We can use an image folder dataset the way we have it setup.
# Create the dataset
 
nz = int(arg_parse.opt.nz) # number of latent variables
ngf = int(arg_parse.opt.ngf) # inside generator
ndf = int(arg_parse.opt.ndf) # inside discriminator
nc = 3 # channels

# custom weights initialization called on netG and netD


#netG = gan_body._netG()
#netG.apply(weights_init)
#if opt.netG != '':
#    netG.load_state_dict(torch.load(opt.netG))
#print(netG)


#netD = gan_body._netD()
#Guiqui 8 layers version
#netD = gan_body._netD_8()

#Guiqiu Resnet version

netD = Net_pair_path_body  ._pair_2_pathNet_( None)
gancreator = cGAN_build2.CGAN_creator() # the Cgan for the segmentation 
GANmodel= gancreator.creat_cgan()  #  G and D are created here 
#netD = gan_body._netD_Resnet()
GANmodel.netG.apply(weights_init)
GANmodel.netG.cuda()

if Continue_flag == True:
    #netD.load_state_dict(torch.load(opt.netD))
    GANmodel.netG.load_state_dict(torch.load(pth_save_dir+'cGANG_epoch_5.pth'))
    #GANmodel.netD.load_state_dict(torch.load(pth_save_dir+'cGAND_epoch_1.pth'))
    #GANmodel.netG.side_branch1. load_state_dict(torch.load(pth_save_dir+'cGANG_branch1_epoch_1.pth'))
print(GANmodel.netG)


netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))

print(netD)
 
 


#criterion = nn.BCELoss()
#criterion = nn.MSELoss()
criterion = nn.L1Loss()

#criterion = nn.CrossEntropyLoss()

input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
label = torch.FloatTensor(opt.batchSize)
real_label = 1
fake_label = 0

if opt.cuda:
    print("CUDA TRUE")
    netD.cuda()
    #netG.cuda()
    criterion.cuda()
    input, label = input.cuda(), label.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()


fixed_noise = Variable(fixed_noise)

# setup optimizer
#optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
#optimizerD = optim.SGD(netD.parameters(), lr=opt.lr,momentum= 0.9, weight_decay =2e-4 )


#saved_stastics = MY_ANALYSIS()
#saved_stastics=saved_stastics.read_my_signal_results()
#saved_stastics.display()

read_id =0

epoch=0
#transform = BaseTransform(  Resample_size,(104/256.0, 117/256.0, 123/256.0))
#transform = BaseTransform(  Resample_size,[104])  #gray scale data
iteration_num =0
mydata_loader = myDataloader_for_shift (Batch_size,Resample_size,Path_length)
multi_scale_weight = [0.005, 0.01, 0.02, 0.08, 0.32]
#multi_scale_weight = [0.32, 0.08, 0.02, 0.01, 0.005]
#multi_scale_weight = [0.2, 0.2, 0.2, 0.2, 0.2]


while(1):
    epoch+= 1
    #almost 900 pictures
    while(1):
        iteration_num +=1
        read_id+=1
        if (mydata_loader.read_all_flag ==1):
            read_id =0
            mydata_loader.read_all_flag =0
            break


        mydata_loader.read_a_batch()
        #change to 3 chanels
        #ini_input = mydata_loader.input_mat
        #np_input = numpy.append(ini_input,ini_input,axis=1)
        #np_input = numpy.append(np_input,ini_input,axis=1)
        ini_input_pair1 = mydata_loader.input_pair1
        ini_input_pair2 = mydata_loader.input_pair2
        ini_input_0   = ini_input_pair2
        np_input = numpy.append(mydata_loader.input_pair3,mydata_loader.input_pair4,axis=1)
        #np_input = numpy.append(np_input,mydata_loader.input_pair3,axis=1)
        #np_input = numpy.append(np_input,mydata_loader.input_pair4,axis=1)


        #np_input = numpy.append(np_input,ini_input_pair2,axis=1)

        input = torch.from_numpy(numpy.float32(np_input)) 
        #input = input.to(device) 
        #input = torch.from_numpy(numpy.float32(mydata_loader.input_image[0,:,:,:])) 
        input = input.to(device)                
   
        patht= torch.from_numpy(numpy.float32(mydata_loader.input_path) )
        #patht=patht.to(device)
                
        #patht= torch.from_numpy(numpy.float32(mydata_loader.input_path[0,:])/71.0 )
        patht=patht.to(device)
        #inputv = Variable(input)
        # using unsqueeze is import  for with out bactch situation
        #inputv = Variable(input.unsqueeze(0))

        #labelv = Variable(patht)
        #inputv = input

        #labelv = patht
        inputv = Variable(input )
        #inputv = Variable(input.unsqueeze(0))
        patht =patht. unsqueeze(1)

        target = Variable(patht )


        #output = netD(inputv)
        GANmodel.update_learning_rate()    # update learning rates in the beginning of every epoch.
        GANmodel.set_input( target,inputv)         # unpack data from dataset and apply preprocessing
        GANmodel.optimize_parameters()


        #loss = multiscaleEPE(output, target, multi_scale_weight, False)

        #output = output.view(Batch_size,Path_length).squeeze(1)
        save_out  = GANmodel.out_pathes0
        #save_out  = GANmodel.path_long1

  

        #errD_real = criterion(output, labelv)
        #errD_real.backward()
        #loss.backward()

       
        # train with fake
        if cv2.waitKey(12) & 0xFF == ord('q'):
              break 
         
        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, 0, read_id, 0,
                 0, 0, 0, 0, 0))
        if Visdom_flag == True:
                plotter.plot( 'l0', 'l0', 'l0', iteration_num, GANmodel.displayloss0.cpu().detach().numpy())

                plotter.plot( 'l1', 'l1', 'l1', iteration_num, GANmodel.displayloss1.cpu().detach().numpy())
                plotter.plot( 'l2', 'l2', 'l2', iteration_num, GANmodel.displayloss2.cpu().detach().numpy())
                plotter.plot( 'l3', 'l3', 'l3', iteration_num, GANmodel.displayloss3.cpu().detach().numpy())

        if read_id % 2 == 0:
            #vutils.save_image(real_cpu,
            #        '%s/real_samples.png' % opt.outf,
            #        normalize=True)
            #netG.eval()
            #fake = netG(fixed_noise)
            #cv2.imwrite(Save_pic_dir  + str(i) +".jpg", mat)
            #show the result

            dispay_id =0
            Matrix  =   mydata_loader.input_mat[dispay_id,0,:,:] +104
            show1 = Matrix*0
            path2 = (mydata_loader.input_path[dispay_id,:])*Original_window_Len
            
            save_out = save_out[0] 
            save_out = save_out[dispay_id] 
            save_out_long = save_out 
            save_out_long = save_out_long.cpu().detach().numpy()


            save_out_long = save_out_long * Original_window_Len
            save_out  = (save_out.data.mean()) *(Original_window_Len)  
            path2_short  = signal.resample(path2, len(save_out_long))

            for i in range ( len(path2_short)):
                path2_short[i] = min(path2_short[i],  Original_window_Len-1)
                path2_short[i] = max(path2_short[i],0)
                save_out_long[i] = min(save_out_long[i],  Original_window_Len-1)
                save_out_long[i] = max(save_out_long[i],0)
                show1[int(path2_short[i]),i]=254
                show1[int(save_out_long[i]),i]=128

  
            save_out_long  = signal.resample(save_out_long, Resample_size)

            #save_out  = signal.resample(save_out, Mat_size)
 
            show2  = Matrix*0
            
            #if save_out >= Original_window_Len:
            #    save_out  = Original_window_Len-1
            #show2[int(save_out ),:]=254
            for i in range ( len(path2)):
                path2[i] = min(path2[i],  Original_window_Len-1)
                path2[i] = max(path2[i],0)
                save_out_long[i] = min(save_out_long[i],  Original_window_Len-1)
                save_out_long[i] = max(save_out_long[i],0)
                show2[int(save_out_long[i]),i]=254
                show2[int(path2[i]),i]=128


            
            show3 = numpy.append(show1,show2,axis=1) # cascade
            cv2.imshow('Deeplearning one',show3.astype(numpy.uint8)) 
            show4 =  mydata_loader.input_pair3[dispay_id,0,:,:]
            show5 =  mydata_loader.input_pair4[dispay_id,0,:,:]
            show6 = numpy.append(show4,show5,axis=0) +104 # cascade
            cv2.imshow('Input one',show6.astype(numpy.uint8)) 

            if cv2.waitKey(12) & 0xFF == ord('q'):
              break
    # do checkpointing
    #torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
    #torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch
    # torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))
    #torch.save(netD.state_dict(), pth_save_dir+ "netD_epoch_"+str(epoch)+".pth")
    torch.save(GANmodel.netG.state_dict(), pth_save_dir+ "cGANG_epoch_"+str(epoch)+".pth")


    #cv2.imwrite(Save_pic_dir  + str(epoch) +".jpg", show2)

    cv2.imwrite(pth_save_dir  + str(epoch) +".jpg", show2)
    if epoch >=5:
        epoch =0


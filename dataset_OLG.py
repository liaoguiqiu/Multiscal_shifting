import cv2
import numpy as np
import os
from analy import MY_ANALYSIS
from analy import Save_signal_enum
from image_trans import BaseTransform  
import random
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import scipy.signal as signal

Batch_size = 10
Resample_size =512
Resample_size2 = 200
Path_length = 1
Mat_size   = 71
Original_window_Len  = 71
transform_img = BaseTransform(  Resample_size,[104])  #gray scale data
transform_mat = BaseTransform(  Mat_size,[104])  #gray scale data
Crop_start = 0
Crop_end  = 200
Display_nurd_flag = False
class myDataloader_for_shift_OLG(object):
    def __init__(self, batch_size,image_size,path_size):
        self.random_shihft_flag  =True # this is used  for add tiny shihft to on line augment the iamge 
        self.data_origin = "../dataset/For_shift_train/saved_original_for_generator/"  # assume this one is the newest frame

        self.data_pair1_root = "../dataset/For_shift_train/pair1/"  # assume this one is the newest frame
        self.data_pair2_root = "../dataset/For_shift_train/pair2/" # assume this one is the historical image
        self.data_mat_root = "../dataset/For_shift_train/CostMatrix/"
        self.signalroot ="../dataset/For_shift_train/saved_stastics/" 
        self.read_all_flag=0
        self.read_record =0
        self.folder_pointer = 0
        self.slice_record=1
        self.batch_size  = batch_size
        self.img_size  = Resample_size
        self.path_size  = Path_length
        self.mat_size  = Mat_size
        self.img_size2  = Resample_size2
        self.W =832
        self.H = 1024

        # Initialize the inout for the tainning
        self.input_mat = np.zeros((batch_size,1,Mat_size,Resample_size)) #matri
        self.input_path = np.zeros((batch_size,Path_length))#path
        self.input_pair1 = np.zeros((batch_size,1,Resample_size2,Resample_size))#pairs
        self.input_pair2 = np.zeros((batch_size,1,Resample_size2,Resample_size))
        self.input_pair3 = np.zeros((batch_size,1,Resample_size2,Resample_size))
        self.input_pair4 = np.zeros((batch_size,1,Resample_size2,Resample_size))

        # the number isdeter by teh mat num
        self.all_dir_list = os.listdir(self.data_mat_root)
        self.folder_num = len(self.all_dir_list)
        # create the buffer list(the skill to create the list)
        self.folder_mat_list = [None]*self.folder_num
        self.folder_pair1_list = [None]*self.folder_num
        self.folder_pair2_list = [None]*self.folder_num
        self.signal = [None]*self.folder_num

        # create all  the folder list and their data list

        number_i = 0
        # all_dir_list is subfolder list 
        #creat the image list point to the STASTICS TIS  list
        saved_stastics = MY_ANALYSIS()
        #read all the folder list of mat and pairs and path
        for subfold in self.all_dir_list:
            #the mat list
            this_folder_list =  os.listdir(os.path.join(self.data_mat_root, subfold))
            this_folder_list2 = [ self.data_mat_root +subfold + "/" + pointer for pointer in this_folder_list]
            self.folder_mat_list[number_i] = this_folder_list2

            #the pair1 list
            this_folder_list =  os.listdir(os.path.join(self.data_pair1_root, subfold))
            this_folder_list2 = [ self.data_pair1_root +subfold + "/" + pointer for pointer in this_folder_list]
            self.folder_pair1_list[number_i] = this_folder_list2
            #the pair2 list
            this_folder_list =  os.listdir(os.path.join(self.data_pair2_root, subfold))
            this_folder_list2 = [ self.data_pair2_root +subfold + "/" + pointer for pointer in this_folder_list]
            self.folder_pair2_list[number_i] = this_folder_list2
            #the supervision signal list
               #change the dir firstly before read
            saved_stastics.all_statics_dir = os.path.join(self.signalroot, subfold, 'signals.pkl')
            self.signal[number_i]  =  saved_stastics.read_my_signal_results()
            
            number_i +=1
            #read the folder list finished  get the folder list and all saved path
    def gray_scale_augmentation(self,orig_gray,amplify_value) :
        random_scale = 0.7 + (1.3  - 0.7) *amplify_value
        aug_gray = orig_gray * random_scale
        aug_gray = np.clip(aug_gray, a_min = 1, a_max = 254)

        return aug_gray
    def noisy(self,noise_typ,image):
           if noise_typ == "gauss_noise":
              row,col = image.shape
              mean = 0
              var = 50
              sigma = var**0.5
              gauss = np.random.normal(mean,sigma,(row,col )) 
              gauss = gauss.reshape(row,col ) 
              noisy = image + gauss
              return np.clip(noisy,0,254)
           elif noise_typ == 's&p':
              row,col  = image.shape
              s_vs_p = 0.5
              amount = 0.004
              out = np.copy(image)
              # Salt mode
              num_salt = np.ceil(amount * image.size * s_vs_p)
              coords = [np.random.randint(0, i - 1, int(num_salt))
                      for i in image.shape]
              out[coords] = 1

              # Pepper mode
              num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
              coords = [np.random.randint(0, i - 1, int(num_pepper))
                      for i in image.shape]
              out[coords] = 0
              return np.clip(out,0,254)
           elif noise_typ == 'poisson':
              vals = len(np.unique(image))
              vals = 2 ** np.ceil(np.log2(vals))
              noisy = np.random.poisson(image * vals) / float(vals)
              return np.clip(noisy,0,254)
           elif noise_typ =='speckle':
              row,col  = image.shape
              gauss = np.random.randn(row,col )
              gauss = gauss.reshape(row,col )        
              noisy = image + image * gauss
              return np.clip(noisy,0,254)
    def small_random_shift(self,orig_gray,random_1) :
        random_shift  = 10 * (random_1 - 0.5)
         
        shifted = np.roll(orig_gray, int(random_shift), axis = 1)     # Positive x rolls right

        return shifted 
    def image3_append(self,img):
        long = np.append(img,img,axis=1)
        long = np.append(long,img,axis=1)
        return long
    def image2_append(self,img):
        long = np.append(img,img,axis=1)
        #long = np.append(long,img,axis=1)
        return long
    # let read a bathch
    # the bis is removed already 
    def de_distortion_integral(self,image,shift_integral ):
        new= image
        h, w= image.shape
        new=new*0  # Nan pixel will be filled by intepolation processing
        mask =  new  + 255
       

        #shift_integral = shift_integral + shift_diff.astype(int) # not += : this is iteration way
        #shift_integral = np.clip(shift_integral, - 35,35)
        #every line will be moved to a new postion
        for i in range ( len(shift_integral)):
            #limit the integral             
            new_position  = int(shift_integral[i]+i)
            # deal with the boundary exceeding
            if(new_position<0):
                new_position = w+new_position
            if (new_position>=w):
                new_position= new_position -w
            #move this line to new position
            new[:,int(new_position)] = image[:,i]
            mask[:,int(new_position)] = 0

        # connect the statrt and end before the interpilate          
        #modified to  # connect the statrt and end before the interpilate
        long_3_img  = np.append(new,new,axis=1) 
        long_3_img = np.append(long_3_img,new,axis=1) # cascade
        longmask  = np.append(mask,mask,axis=1) 
        longmask  = np.append(longmask,mask,axis=1) 

        # interp_img=cv2.inpaint(long_3_img, longmask, 2, cv2.INPAINT_TELEA)
        interp_img=cv2.inpaint(new, mask, 1, cv2.INPAINT_TELEA)
        # the time cmcumption of this is 0.02s
        

        #interp_img = VIDEO_PEOCESS.img_interpilate(long_3_img) # interpolate by row
        # new= interp_img[:,w:2*w] # take the middle one 
        new= interp_img  

        return new
    def read_a_batch(self):
        read_start = self.read_record
        #read_end  = self.read_record+ self.batch_size
        thisfolder_len =  len (self.folder_pair1_list[self.folder_pointer])
        
            #return self.input_mat,self.input_path# if out this folder boundary, just returen
        this_pointer=0
        i=read_start
        while (1):
        #for i in range(read_start, read_end):
            #this_pointer = i -read_start
            # get the all the pointers 
            #Image_ID , b = os.path.splitext(os.path.dirname(self.folder_list[self.folder_pointer][i]))

            Path_dir,Image_ID =os.path.split(self.folder_mat_list[self.folder_pointer][i])
            Image_ID_str,jpg = os.path.splitext(Image_ID)
            Image_ID = int(Image_ID_str)
            #start to read image and paths to fill in the input bach
            this_mat_path = self.folder_mat_list[self.folder_pointer][i] # read saved mat
            this_mat = cv2.imread(this_mat_path)

            # ramdon select origina
            OriginalpathDirlist = os.listdir(self.data_origin) 
            sample = random.sample(OriginalpathDirlist, 1)  #  ramdom choose the name in folder list
            Sample_path = self.data_origin +   sample[0] # create the reading path this radom picture
            original_IMG = cv2.imread(Sample_path) # get this image 
            original_IMG  =   cv2.cvtColor(original_IMG, cv2.COLOR_BGR2GRAY) # to gray
            original_IMG = cv2.resize(original_IMG, (self.W,self.H), interpolation=cv2.INTER_AREA)


            #this_pair1_path = self.folder_pair1_list[self.folder_pointer][i] # read saved pair1
            #this_pair1 = cv2.imread(this_pair1_path)
            #this_pair2_path = self.folder_pair2_list[self.folder_pointer][i] # read saved pair1
            #this_pair2 = cv2.imread(this_pair2_path)
            #resample 
            #this_img = cv2.resize(this_img, (self.img_size,self.img_size), interpolation=cv2.INTER_AREA)
           
            #get the index of this Imag path
            #Path_Index_list = self.signal[self.folder_pointer].signals[Save_signal_enum.image_iD.value,:]
            #Path_Index_list = Path_Index_list.astype(int)
            #Path_Index_list = Path_Index_list.astype(str)

            #try:
            #    Path_Index = Path_Index_list.tolist().index(Image_ID_str)
            #except ValueError:
            #    print(Image_ID_str + "not path exsting")

            #else:             
            #Path_Index = Path_Index_list.tolist().index(Image_ID_str)            
            #this_path = self.signal[self.folder_pointer].path_saving[Path_Index]

            random_NURD   = np.random.random_sample(20)*10
            random_NURD  = signal.resample(random_NURD, self.W)

            random_NURD = gaussian_filter1d(random_NURD,10) # smooth the path 
            random_shifting = np.random.random_sample()*Original_window_Len
            random_NURD= random_NURD- np.mean(random_NURD) + random_shifting  
            if Display_nurd_flag == True:
                fig = plt.figure()
                ax = plt.axes()
                ax.plot(random_NURD)
            random_NURD = random_NURD  -  Original_window_Len/2

            #path2 =  signal.resample(this_path, self.path_size)#resample the path
            # concreate the image batch and path
            this_mat  =   cv2.cvtColor(this_mat, cv2.COLOR_BGR2GRAY)
            this_pair1  =   original_IMG
            this_pair2  =   self.de_distortion_integral (this_pair1,random_NURD)
            noise_selector=['gauss_noise','speckle','gauss_noise','speckle']
            noise_it = np.random.random_sample()*5
            noise_type1  =  str( noise_selector[int(noise_it)%4])
            noise_it = np.random.random_sample()*5
            noise_type2  =  str( noise_selector[int(noise_it)%4])
            #noise_type = "gauss_noise"
            this_pair1  =  self.noisy(noise_type1,this_pair1)
            this_pair2  =  self.noisy(noise_type2,this_pair2)
 

            #this_mat = self.gray_scale_augmentation(this_mat,amplifier)
            H_mat,W_mat = this_mat.shape
            H_img,W_img = this_pair1.shape
            this_mat = cv2.resize(this_mat, (Resample_size,Mat_size), interpolation=cv2.INTER_AREA)
                 
                 
                 
            # imag augmentation
            amplifier1  = np.random.random_sample()
            amplifier2  = np.random.random_sample()
            amplifier3  = np.random.random_sample()

            pair1_piece =   self.gray_scale_augmentation(this_pair1[Crop_start:Crop_end,:],amplifier1)
            pair2_piece =   self.gray_scale_augmentation(this_pair2[Crop_start:Crop_end,:],amplifier2)
            pair3_piece =   self.gray_scale_augmentation(this_pair1 ,amplifier1)
            pair4_piece =   self.gray_scale_augmentation(this_pair2,amplifier3)

            if (self.random_shihft_flag == True):
                delta  = np.random.random_sample()
                pair4_piece  = self.small_random_shift(pair4_piece,delta)


            pair1_piece  =  cv2.resize(self.image2_append(pair1_piece), (Resample_size,Resample_size2), interpolation=cv2.INTER_AREA)
            pair2_piece  =  cv2.resize(self.image2_append(pair2_piece), (Resample_size,Resample_size2), interpolation=cv2.INTER_AREA)
            pair3_piece  =  cv2.resize(self.image2_append(pair3_piece), (Resample_size,Resample_size2), interpolation=cv2.INTER_AREA)
            pair4_piece  =  cv2.resize(self.image2_append(pair4_piece), (Resample_size,Resample_size2), interpolation=cv2.INTER_AREA)
            #fill in the batch
            self.input_mat[this_pointer,0,:,:] = this_mat #transform_mat(this_ma)[0]
            #self.input_pair1[this_pointer,0,:,:] = transform_img(pair1_piece)[0]
            #self.input_pair2[this_pointer,0,:,:] = transform_img(pair2_piece)[0]
            self.input_pair1[this_pointer,0,:,:] = pair1_piece -104
            self.input_pair2[this_pointer,0,:,:] = pair2_piece - 104
            self.input_pair3[this_pointer,0,:,:] = pair3_piece -104
            self.input_pair4[this_pointer,0,:,:] = pair4_piece - 104
            self.input_path [this_pointer , :] = random_shifting / Original_window_Len
            this_pointer +=1
 

            i+=1
            if (i>=thisfolder_len):
                i=0
                self.read_record =0
                self.folder_pointer+=1
                if (self.folder_pointer>= self.folder_num):
                    self.read_all_flag =1
                    self.folder_pointer =0
            if(this_pointer>=self.batch_size): # this batch has been filled
                break
            pass
        self.read_record=i # after reading , remember to  increase it 
        return self.input_mat,self.input_path


##test read 
#data  = myDataloader (Batch_size,Resample_size,Path_length)

#for  epoch in range(500):

#    while(1):
#        data.read_a_batch()



expr_dir = '/home/azhar/crnn-pytorch/output_run3/' # where to store samples and models
crnn_path='/home/azhar/crnn-pytorch/output_run1/netCRNNbest.pth'#path to trained recognition model
PSEnet_path='/home/azhar/summerResearch/MyPSEnet/0.234_epoch6_checkpoint.pth.tar.part'#path to trained detection model

expr_dir = './output'
crnn_path = './models/netCRNNbest.pth'
PSEnet_path = './models/PSEnet_best.pth.tar.part'

# about data and net
alphabet = alpha='.:ँंःअआइईउऊऋएऐऑओऔकखगघचछजझञटठडढणतथदधनपफबभमयरलळवशषसहािीुूृॅेैॉोौ्ॐड़ढ़०१२३४५६७८९\u200c\u200d()'
keep_ratio = False # whether to keep ratio for image resize
manualSeed = 1234 # reproduce experiemnt
random_sample = True # whether to sample the dataset with random sampler
imgH = 32 # the height of the input image to network
imgW = 100 # the width of the input image to network
nh = 256 # size of the lstm hidden state
nc = 1
dealwith_lossnone = True # whether to replace all nan/inf in gradients to zero
# hardware
cuda = True # enables cuda
multi_gpu = False # whether to use multi gpu
ngpu = 1 # number of GPUs to use. Do remember to set multi_gpu to True!
workers = 0 # number of data loading workers

# training process
displayInterval = 10 # interval to be print the train loss
valInterval = 2 # interval to val the model loss and accuray
saveInterval = 4 # interval to save model
n_test_disp = 10 # number of samples to display when val the model

# finetune
nepoch = 1000 # number of epochs to train for
batchSize = 16 # input batch size
lr = 0.01 # learning rate for Critic, not used by adadealta
beta1 = 0.5 # beta1 for adam. default=0.5
adam = False # whether to use adam (default is rmsprop)
adadelta = True # whether to use adadelta (default is rmsprop)

#---------------------------Parameters for PSEnet------------------------------------------------
arch='resnet50'#specify architecture
binary_th=1.0
kernel_num=7
scale=1
long_size=2240
min_kernel_area=5.0
min_area=800.0
min_score=0.93

from easydict import EasyDict 

OPTION = EasyDict()

# ------------------------------------------ data configuration ---------------------------------------------
OPTION.trainset = ['DAVIS17','VOS']
OPTION.valset = 'DAVIS17'
OPTION.datafreq = [5, 1]
OPTION.input_size = (240, 427) #(320, 570) (240, 427)   # input image size
OPTION.sampled_frames = 3        # min sampled time length while trianing
OPTION.max_skip = [5, 3]         # max skip time length while trianing
OPTION.samples_per_video = 2    # sample numbers per video
OPTION.with_coco = True
OPTION.coco_ratio = 0.01

# ----------------------------------------- model configuration ---------------------------------------------
OPTION.keydim = 128
OPTION.valdim = 512
OPTION.save_freq = 5
OPTION.epochs_per_increment = 5

OPTION.backbone = 'resnet34' #'resnet34' or 'resnet50'
OPTION.layer = 'r2' # r1,r2,r3,r4
OPTION.fusion_type = 'se' # 'se' or 'add'
OPTION.bg_type = 'concat' # '', 'concat', 'branch', 'conv1'
OPTION.pred_layer = 'r4' # r4, r5, mask

# ---------------------------------------- training configuration -------------------------------------------
OPTION.epochs = 130
# OPTION.epochs = 12
OPTION.train_batch = 8
OPTION.workers = 8
OPTION.learning_rate = 0.00001
OPTION.gamma = 0.1
OPTION.momentum = (0.9, 0.999)
OPTION.solver = 'adam'             # 'sgd' or 'adam'
OPTION.weight_decay = 5e-4
OPTION.iter_size = 1
OPTION.milestone = []              # epochs to degrades the learning rate
OPTION.loss = 'both'               # 'ce' or 'iou' or 'both'
OPTION.mode = 'recurrent'          # 'mask' or 'recurrent' or 'threshold'
OPTION.iou_threshold = 0.65        # used only for 'threshold' training
OPTION.save_model_freq = 5         # frequence for saving model
OPTION.correction_iter_times = 10
OPTION.correction_lr = 150
OPTION.memory_size = 1
OPTION.fix_mem = True
OPTION.add_thres = 0.85
OPTION.loss_coeff = (1, 1)
OPTION.exp_name = 'fast'

# ---------------------------------------- testing configuration --------------------------------------------
OPTION.epoch_per_test = OPTION.epochs
OPTION.save_indexed_format = True # set True to save indexed format png file, otherwise segmentation with original image
# OPTION.results = '/public/home/jm/Data/output/tmp/results_{}/'.format(OPTION.exp_name)
OPTION.results = 'output/results/{}'.format(OPTION.exp_name)
OPTION.gpu_id = '0'      # defualt gpu-id (if not specified in cmd)

# ------------------------------------------- other configuration -------------------------------------------
OPTION.pretrained = '/public/home/jm/Data/output/models_{}/'.format(OPTION.exp_name)
OPTION.initial = ''      # path to initialize the backbone
OPTION.checkpoint = '/public/home/jm/Data/output/models_{}/'.format(OPTION.exp_name) 
OPTION.resume = ''
# OPTION.resume = '/public/home/jm/Data/output/stm_output/models_{}/DAVIS17/recurrent.pth.tar'.format(OPTION.exp_name) # path to restart from the checkpoint

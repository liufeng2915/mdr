import os
from datetime import datetime
from pyhocon import ConfigFactory
import sys
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import numpy as np  
import scipy.io
from torch.nn import functional as F
from tensorboardX import SummaryWriter
from torch.autograd import Variable

import utils.general as utils
import utils.plots as utils_plt
import utils.data as utils_data
import utils.debug as utils_debug

class TrainRunner():
    def __init__(self,**kwargs):
        torch.set_default_dtype(torch.float32)
        torch.set_num_threads(1)

        self.conf = ConfigFactory.parse_file(kwargs['conf'])
        self.batch_size = kwargs['batch_size']
        self.nepochs = kwargs['nepochs']
        self.exps_folder_name = kwargs['exps_folder_name']
        self.GPU_INDEX = kwargs['gpu_index']
        self.expname = self.conf.get_string('train.expname')
        self.device = torch.device(self.GPU_INDEX)
        print(self.device)

        # # pca model, latent
        mat_file = scipy.io.loadmat('../data/pca_model.mat')
        pca_base = mat_file["base"].astype(np.float32)
        pca_base = pca_base[:self.conf.get_int('model.recon_network.feat_dim')]
        pca_mean = mat_file["mu"].astype(np.float32)
        mat_file = scipy.io.loadmat('../data/mean_std_latent.mat')
        mean_latent = mat_file["mean_latent"].astype(np.float32)
        std_latent = mat_file["std_latent"].astype(np.float32)
        self.pca_model = {
            "pca_base": pca_base,
            "pca_mean": pca_mean,
            "mean_latent": mean_latent,
            "std_latent": std_latent
        }


        if kwargs['is_continue'] and kwargs['timestamp'] == 'latest':
            if os.path.exists(os.path.join('../',kwargs['exps_folder_name'],self.expname)):
                timestamps = os.listdir(os.path.join('../',kwargs['exps_folder_name'],self.expname))
                if (len(timestamps)) == 0:
                    is_continue = False
                    timestamp = None
                else:
                    timestamp = sorted(timestamps)[-1]
                    is_continue = True
            else:
                is_continue = False
                timestamp = None
        else:
            timestamp = kwargs['timestamp']
            is_continue = kwargs['is_continue']

        utils.mkdir_ifnotexists(os.path.join('../',self.exps_folder_name))
        self.expdir = os.path.join('../', self.exps_folder_name, self.expname)
        utils.mkdir_ifnotexists(self.expdir)
        self.timestamp = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())
        utils.mkdir_ifnotexists(os.path.join(self.expdir, self.timestamp))

        self.plots_dir = os.path.join(self.expdir, self.timestamp, 'plots')
        utils.mkdir_ifnotexists(self.plots_dir)

        # create checkpoints dirs
        self.checkpoints_path = os.path.join(self.expdir, self.timestamp, 'checkpoints')
        utils.mkdir_ifnotexists(self.checkpoints_path)
        self.model_params_subdir = "ModelParameters"
        self.optimizer_params_subdir = "OptimizerParameters"
        self.scheduler_params_subdir = "SchedulerParameters"
        self.log_subdir = "logs"

        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.model_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.optimizer_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.scheduler_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.log_subdir))

        os.system("""cp -r {0} "{1}" """.format(kwargs['conf'], os.path.join(self.expdir, self.timestamp, 'runconf.conf')))

        if (not self.GPU_INDEX == 'ignore'):
            os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(self.GPU_INDEX)

        print('shell command : {0}'.format(' '.join(sys.argv)))

        # # load data
        print('Loading data ...')
        self.dataset_conf = self.conf.get_config('dataset')
        self.train_dataset = utils.get_class(self.conf.get_string('train.dataset_class'))(is_train=True,
                                                                                          pca_data=self.pca_model, 
                                                                                          **self.dataset_conf)
        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                            num_workers=4,
                                                            drop_last=True,
                                                            batch_size=self.batch_size,
                                                            shuffle=True,
                                                            collate_fn=utils_data.collator
                                                            )
        self.plot_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                           num_workers=1,
                                                           batch_size=self.conf.get_int('plot.plot_nimgs'),
                                                           shuffle=True,
                                                           collate_fn=utils_data.collator
                                                           )

        # # define model
        self.model = utils.get_class(self.conf.get_string('train.model_class'))(conf=self.conf)
        if torch.cuda.is_available():
            self.model.to(self.device)

        # # define loss
        self.loss = utils.get_class(self.conf.get_string('train.loss_class'))(self.device,
                                                                              self.batch_size,
                                                                              self.conf.get_int('dataset.num_train_voxel_sdf'), 
                                                                              self.pca_model, 
                                                                              self.conf.get_list('dataset.dim_reference'),
                                                                              self.conf.get_float('dataset.voxel_size'), 
                                                                              **self.conf.get_config('loss'))

        # # learning rate
        self.lr = self.conf.get_float('train.learning_rate')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.sched_milestones = self.conf.get_list('train.sched_milestones', default=[])
        self.sched_factor = self.conf.get_float('train.sched_factor', default=0.0)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, self.sched_milestones, gamma=self.sched_factor)

        # #
        self.start_epoch = 0
        if is_continue:
            old_checkpoints_dir = os.path.join(self.expdir, timestamp, 'checkpoints')

            saved_model_state = torch.load(
                os.path.join(old_checkpoints_dir, 'ModelParameters', str(kwargs['checkpoint']) + ".pth"))
            self.model.load_state_dict(saved_model_state["model_state_dict"])
            self.start_epoch = saved_model_state['epoch']

            data = torch.load(
                os.path.join(old_checkpoints_dir, 'OptimizerParameters', str(kwargs['checkpoint']) + ".pth"))
            self.optimizer.load_state_dict(data["optimizer_state_dict"])

            data = torch.load(
                os.path.join(old_checkpoints_dir, self.scheduler_params_subdir, str(kwargs['checkpoint']) + ".pth"))
            self.scheduler.load_state_dict(data["scheduler_state_dict"])

        self.plot_freq = self.conf.get_int('train.plot_freq')
        self.plot_nimgs = self.conf.get_int('plot.plot_nimgs')
        self.eval_freq = self.conf.get_int('train.eval_freq')

        # # log
        self.summary = SummaryWriter(os.path.join(self.checkpoints_path, self.log_subdir))


    def save_checkpoints(self, epoch):

        torch.save(
            {"epoch": epoch, "model_state_dict": self.model.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "model_state_dict": self.model.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, "latest.pth"))

        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.optimizer_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.optimizer_params_subdir, "latest.pth"))

        torch.save(
            {"epoch": epoch, "scheduler_state_dict": self.scheduler.state_dict()},
            os.path.join(self.checkpoints_path, self.scheduler_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "scheduler_state_dict": self.scheduler.state_dict()},
            os.path.join(self.checkpoints_path, self.scheduler_params_subdir, "latest.pth"))


    def run(self):
        print("training...")

        for epoch in range(self.start_epoch, self.nepochs + 1):

            if epoch % 1 == 0:
                self.save_checkpoints(epoch)

            # # training
            for data_index, data in enumerate(self.train_dataloader):   

                file_name = data["img_name"]
                images = data["images"].to(self.device)
                images = utils_data.to_image_list(images).tensors
                targets = [target.to(self.device) for target in data["targets"]]
                K = torch.stack([t.get_field("K") for t in targets])
                RT = torch.stack([t.get_field("RT") for t in targets])

                model_outputs = self.model.forward(K, RT, images)
                loss_output = self.loss.forward(model_outputs, targets)
                loss = loss_output['loss']

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                for key, value in loss_output.items():
                    self.summary.add_scalar('train/loss/{}'.format(key), value, epoch*len(self.train_dataloader)+data_index)

                print(
                    'Time: {0}, Epoch: [{1}]  Data: ({2}/{3}): loss = {4}, box_loss = {5}, hm_loss = {6}, voxel_loss = {7}, feat_loss = {8}, sdf_loss = {9}, lr = {10}'
                    .format(str(datetime.now()), epoch, data_index, len(self.train_dataloader), loss_output['loss'],
                        loss_output['box_loss'],
                        loss_output['hm_loss'],
                        loss_output['voxel_loss'],
                        loss_output['feat_loss'],
                        loss_output['sdf_loss'],
                        self.scheduler.get_last_lr()[0])
                    )

                # # visualization
                if data_index % self.plot_freq == 0:

                    self.save_checkpoints(epoch)
                    save_path = os.path.join(self.plots_dir, str(epoch)+"_"+str(data_index)+"_"+file_name[0])
                    utils.mkdir_ifnotexists(save_path)
                    #
                    utils_plt.visualize_image(images[0], self.dataset_conf, save_path)

                    ## esti
                    utils_plt.visualize_hm(model_outputs["predict_class"][0], model_outputs["visible"][0], save_path, 'esti')
                    utils_plt.visualize_voxe(model_outputs["predict_voxel"][0], model_outputs["visible"][0], save_path, 'esti')

                    ## gt
                    gt_heatmaps = torch.stack([t.get_field("hm") for t in targets])[0]
                    gt_voxel = torch.stack([t.get_field("voxel") for t in targets])[0]
                    utils_plt.visualize_hm(gt_heatmaps, model_outputs["visible"][0], save_path, 'gt')
                    utils_plt.visualize_voxe(gt_voxel, model_outputs["visible"][0], save_path, 'gt')

            self.scheduler.step()

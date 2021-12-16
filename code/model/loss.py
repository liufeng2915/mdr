import torch
from torch import nn
from torch.nn import functional as F
import utils.general as utils


class DetReconLoss(nn.Module):
    def __init__(self, device, batch_size, num_train_voxel_sdf, pca_model, dim_reference, grid_size, box_weight, heatmap_weight, voxel_weight, sdf_feat_weight, sdf_weight, alpha, beta):
        super().__init__()

        self.pca_base = torch.FloatTensor(pca_model["pca_base"]).to(device)
        self.pca_mean = torch.FloatTensor(pca_model["pca_mean"]).to(device)
        self.mean_latent = torch.FloatTensor(pca_model["mean_latent"]).to(device)
        self.std_latent = torch.FloatTensor(pca_model["std_latent"]).to(device)
        self.mean_latent = self.mean_latent.unsqueeze(1).repeat(1,num_train_voxel_sdf,1)
        self.std_latent = self.std_latent.unsqueeze(1).repeat(1,num_train_voxel_sdf,1)
        self.pca_base = self.pca_base.unsqueeze(0).repeat(batch_size,1,1)
        self.pca_mean = self.pca_mean.unsqueeze(0).repeat(batch_size,1,1)

        self.dim_reference = torch.as_tensor(dim_reference)
        self.grid_size = torch.as_tensor(grid_size)

        self.box_weight = box_weight
        self.heatmap_weight = heatmap_weight
        self.voxel_weight = voxel_weight
        self.sdf_feat_weight = sdf_feat_weight
        self.sdf_weight = sdf_weight
        self.alpha = alpha 
        self.beta = beta

        self.l1_loss = nn.L1Loss(reduction='sum')
        self.l1_loss_mean = nn.L1Loss(reduction='mean')
        self.mse_loss = nn.MSELoss(reduction='sum')
        self.mse_loss_mean = nn.MSELoss(reduction='mean')
        self.bce_loss = nn.BCELoss(reduction='sum')

    def get_box_loss(self, esti_box, gt_box, mask, num_objs):

        box_loss = self.l1_loss(esti_box*mask, gt_box*mask) / (self.box_weight * num_objs)
        return box_loss

    def get_heatmap_loss(self, prediction, target, mask, weight):

        alpha = self.alpha
        beta = self.beta

        positive_index = target.eq(1).float()
        negative_index = target.lt(1).float()

        negative_weights = torch.pow(1 - target, beta)
        loss = 0.

        positive_loss = torch.log(prediction) \
        * torch.pow(1 - prediction, alpha) * positive_index
        negative_loss = torch.log(1 - prediction) \
        * torch.pow(prediction, alpha) * negative_weights * negative_index

        num_positive = positive_index.float().sum()
        positive_loss = positive_loss*mask
        negative_loss = negative_loss*mask
        positive_loss = positive_loss.sum()
        negative_loss = negative_loss.sum()

        if num_positive == 0:
            loss -= negative_loss
        else:
            loss -= (positive_loss + negative_loss) / num_positive

        return loss*weight

    def get_sdf_feat_loss(self, esti_feat, gt_feat, weight):

        feat_loss = self.l1_loss_mean(esti_feat, gt_feat)

        return feat_loss*weight

    def get_sdf_loss(self, esti_sdf, gt_sdf, mask, weight):

        sdf_mask = mask.unsqueeze(-1).expand_as(gt_sdf)
        feat_loss = self.mse_loss(esti_sdf*sdf_mask, gt_sdf*sdf_mask)/(torch.nonzero(sdf_mask).shape[0] + 1e-6)

        return feat_loss*weight

    def get_voxel_loss(self, esti_voxel, gt_voxel, mask, weight):

        batch, channel, W, H, D = mask.shape
        voxel_mask = mask.reshape(batch,channel,W*H*D).permute(0,2,1)

        voxel_loss = self.bce_loss(esti_voxel*voxel_mask, gt_voxel*voxel_mask)/(torch.nonzero(voxel_mask).shape[0]+1e-6)

        return voxel_loss*weight


    def forward(self, model_outputs, targets):

        pred_heatmap = model_outputs["predict_class"]
        pred_regression = model_outputs["predict_regression"]
        pred_voxel = model_outputs["predict_voxel"]
        pred_sdf_feat = model_outputs["predict_sdf_feat"]
        visible = model_outputs["visible"].clone()
        device = pred_heatmap.device

        # #
        targets_heatmap, targets_regression, targets_voxel, targets_variables = utils.prepare_targets(targets)
        targets_regression = targets_regression.view(-1, targets_regression.shape[2], targets_regression.shape[3])
        pred_boxes3d = utils.prepare_predictions(self.dim_reference, self.grid_size, targets_variables, pred_regression)

        #
        reg_mask = targets_variables["reg_mask"].flatten()
        num_objs = torch.nonzero(reg_mask).shape[0] + 1e-6
        reg_mask = reg_mask.view(-1, 1, 1)
        reg_mask = reg_mask.expand_as(targets_regression)

        # # box loss
        box_loss_ori = self.get_box_loss(pred_boxes3d["ori_box"], targets_regression, reg_mask, num_objs)
        box_loss_dim = self.get_box_loss(pred_boxes3d["dim_box"], targets_regression, reg_mask, num_objs)
        box_loss_loc = self.get_box_loss(pred_boxes3d["loc_box"], targets_regression, reg_mask, num_objs)
        box_loss = box_loss_ori + box_loss_dim + box_loss_loc

        # # heatmap focal oss
        hm_loss = self.get_heatmap_loss(pred_heatmap, targets_heatmap, visible, self.heatmap_weight)

        # # voxel loss
        voxel_loss = self.get_voxel_loss(pred_voxel, targets_voxel, visible, self.voxel_weight)

        # # feat_loss 
        targets_sdf_feat = targets_variables["sdf_feat"]
        sdf_mask = targets_variables["sdf_mask"]
        sdf_idx = targets_variables["sdf_idx"].long()
        targets_sdf = targets_variables["sdf"]

        pred_pts_feat = pred_sdf_feat.gather(1, sdf_idx.unsqueeze(-1).repeat(1,1,pred_sdf_feat.shape[2]))
        unnorm_pts_feat = pred_pts_feat*self.std_latent + self.mean_latent
        feat_loss = self.get_sdf_feat_loss(unnorm_pts_feat, targets_sdf_feat, self.sdf_feat_weight)

        esti_sdf = torch.einsum('bij,bjk->bik', unnorm_pts_feat, self.pca_base) + self.pca_mean
        sdf_loss = self.get_sdf_loss(esti_sdf, targets_sdf, sdf_mask, self.sdf_weight)

        #----------------------------------------#
        loss = box_loss + hm_loss + voxel_loss + feat_loss + sdf_loss

        return {
            "loss": loss,
            "box_loss": box_loss,
            "box_loss_ori": box_loss_ori,
            "box_loss_dim": box_loss_dim,
            "box_loss_loc": box_loss_loc,
            "hm_loss": hm_loss,
            "voxel_loss": voxel_loss, 
            "feat_loss": feat_loss,
            "sdf_loss": sdf_loss,
            "ori_loss": ori_loss,
        }



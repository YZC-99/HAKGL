import torch
from torch import nn

def KL(out1, out2, eps):
    kl = (out1 * (out1 + eps).log() - out1 * (out2 + eps).log()).sum(dim=1)
    kl = kl.mean()
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(kl)
    return kl

class KGCL(nn.Module):
    def __init__(
        self, knowledge_embeds: torch.Tensor,dataname: str, eps=1e-8, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.loss = nn.CrossEntropyLoss(reduction="none")
        self.knowledge_embeds_T = knowledge_embeds.T
        knowledge_embeds_norm = knowledge_embeds[1:,:] / knowledge_embeds[1:,:].norm(dim=1, keepdim=True)
        self.disease_cosine_sim = torch.mm(knowledge_embeds_norm, knowledge_embeds_norm.t())
        self.eps = eps
        self.dataname = dataname
        self.alpha = 0.6

    def KGCL_inter(self, inputs, targets):
        # inputss: (batch,num_classes,dim)
        diseases_mask = (targets[:, 0] != 1).float()
        if diseases_mask.sum() == 0:
            return torch.tensor(0.0, device=inputs.device)

        diseases_avg_feat_list = []
        if self.dataname == 'odir': # because the first class is normal
            start = 1
        elif self.dataname == 'rfmid' or self.dataname == 'kaggle':
            start = 0
        class_mask = torch.ones(self.disease_cosine_sim.size(0) - start, device=inputs.device)
        for i in range(start, self.disease_cosine_sim.size(0) + start):
            mask = targets[:, i] == 1
            if mask.sum() > 0:
                samples_feat = inputs[mask] # (num_samples, num_classes, dim)
                num_classes_feat = samples_feat[:,i,:] # (num_samples, dim)
                avg_feat = num_classes_feat.mean(dim=0) # (dim,)
                avg_feat = avg_feat / avg_feat.norm()
            else:
                avg_feat = torch.zeros_like(inputs[0,0,:], device=inputs.device)
                class_mask[i - start] = 0
            diseases_avg_feat_list.append(avg_feat)
        diseases_avg_feat = torch.stack(diseases_avg_feat_list)  # shape: (num_diseases, feat_dim)
        diseases_sim = torch.mm(diseases_avg_feat, diseases_avg_feat.t())  # shape: (batch, num_diseases)

        diseases_sim_out = nn.functional.softmax(diseases_sim, dim=0)
        disease_cosine_sim_out = nn.functional.softmax(self.disease_cosine_sim, dim=0)
        loss = KL(diseases_sim_out, disease_cosine_sim_out.detach(), self.eps) + KL(disease_cosine_sim_out.detach(),diseases_sim_out, self.eps)
        return loss

    def KGCL_intra(self, inputs, targets):
        # inputss: (batch,num_classes,dim)
        diseases_mask = (targets[:, 0] != 1).float()
        if diseases_mask.sum() <= 1:
            return torch.tensor(0.0, device=inputs.device)

        multi_label_idx = (targets.sum(dim=1) > 1).nonzero(as_tuple=True)[0]
        all_loss = 0
        for sample_idx in multi_label_idx:
            multi_sample_label = targets[sample_idx]
            multi_sample_feat = inputs[sample_idx]
            valid_idx = (multi_sample_label[1:] == 1).nonzero(as_tuple=True)[0] + 1

            multi_label_feat = multi_sample_feat[valid_idx]
            multi_label_feat = multi_label_feat / multi_label_feat.norm(dim=1, keepdim=True)
            feat_mm = torch.mm(multi_label_feat, multi_label_feat.T)
            # disease_cosine_sim_selected = self.disease_cosine_sim[valid_idx][:, valid_idx]
            disease_idx = valid_idx - 1
            disease_cosine_sim_selected = self.disease_cosine_sim[disease_idx][:, disease_idx]

            feat_out = nn.functional.softmax(feat_mm)
            disease_cosine_sim_out = nn.functional.softmax(disease_cosine_sim_selected)
            sample_loss = KL(feat_out, disease_cosine_sim_out.detach(), self.eps) + KL(disease_cosine_sim_out.detach(),feat_out, self.eps)
            all_loss += sample_loss

        return all_loss / len(multi_label_idx)

    def forward(self, inputs, targets):
        self.knowledge_embeds_T = self.knowledge_embeds_T.to(device=inputs.device, dtype=inputs.dtype)

        return self.alpha * self.KGCL_intra(inputs, targets) + (1 - self.alpha) * self.KGCL_inter(inputs, targets)



class HCCL(nn.Module):

    def __init__(self, tau=0.005, eps=1e-5):
        super(HCCL, self).__init__()
        self.tau = tau
        self.eps = eps
        self.f1=self.f2=self.length_b=self.mean_f1=self.mean_f2=self.length=self.diag=self.new1=self.new2=self.out1=self.out2=None

    def forward(self, feat1, feat2):
        """"
        Parameters
        ----------
        feat1: input features from one stream
        feat2: input features from other stream
        """

        self.f1 = feat1
        self.f2 = feat2

        self.length_b = feat1.size(0)  # batch_size

        self.mean_f1 = torch.reshape(self.f1, (self.length_b, -1))
        self.mean_f2 = torch.reshape(self.f2, (self.length_b, -1))
        self.mean_f1 = nn.functional.normalize(self.mean_f1, 2, dim=1)
        self.mean_f2 = nn.functional.normalize(self.mean_f2, 2, dim=1)

        self.length = self.mean_f1.size(0)
        self.diag = torch.eye(self.length).cuda()
        self.new1 = torch.mm(self.mean_f1, self.mean_f1.t()) / self.tau
        self.new2 = torch.mm(self.mean_f2, self.mean_f2.t()) / self.tau

        self.new1 = self.new1 - self.new1 * self.diag
        self.new2 = self.new2 - self.new2 * self.diag

        self.out1 = self.new1.flatten()[:-1].view(self.length - 1, self.length + 1)[:, 1:].flatten().view(self.length, self.length - 1)  # B*(B-1)
        self.out2 = self.new2.flatten()[:-1].view(self.length - 1, self.length + 1)[:, 1:].flatten().view(self.length, self.length - 1)  # B*(B-1)

        self.out1 = nn.functional.softmax(self.out1)
        self.out2 = nn.functional.softmax(self.out2)

        loss = KL(self.out1, self.out2, self.eps) + KL(self.out2, self.out1, self.eps)
        return loss





if __name__ == '__main__':

    inputs = torch.randn(4, 12, 512).float()
    pairwise_similarity = torch.bmm(inputs, inputs.transpose(1, 2))
    print(pairwise_similarity.shape)
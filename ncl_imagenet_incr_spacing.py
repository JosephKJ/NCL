import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, lr_scheduler
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from utils.util import BCE, PairEnum, cluster_acc, Identity, AverageMeter, seed_torch
from utils import ramps 
from sklearn.cluster import KMeans
from torchvision.models.resnet import BasicBlock
from data.imagenetloader import ImageNetLoader30, ImageNetLoader882_30Mix, ImageNetLoader882, ImageNetLoader30_pre, ImageNetLoader882_30Mix_pre, ImageNetLoader882_pre
from tqdm import tqdm
import numpy as np
import math
import os
import warnings
from models.NCL import NCLMemory
from utils.spacing import CentroidManager
from utils.kmeans import KMeans_cosine_GPU, KMeans_GPU
warnings.filterwarnings("ignore", category=UserWarning)

class ResNet(nn.Module):

    def __init__(self, block, layers, num_labeled_classes=10, num_unlabeled_classes=10):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.head1= nn.Linear(512 * block.expansion, num_labeled_classes)
        self.head2= nn.Linear(512 * block.expansion, num_unlabeled_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, output='None'):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        out = x
        # out = F.relu(out) #add ReLU to benifit ranking
        feat = out
        out1 = self.head1(out)
        feat_norm = F.normalize(out)
        out2 = self.head2(out)
        if output == 'feat_logit':
            return feat, feat_norm, out1, out2
        else:
            return out1, out2


class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        # self.mean = torch.tensor([0.485, 0.456, 0.406]).cuda().view(1, 3, 1, 1)
        self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1, 3, 1, 1)
        # self.std = torch.tensor([0.229, 0.224, 0.225]).cuda().view(1, 3, 1, 1)
        self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1, 3, 1, 1)
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target, self.next_idx = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            self.next_idx = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            self.next_idx = self.next_idx.cuda(non_blocking=True)
            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            self.next_input = self.next_input.float()
            self.next_input = self.next_input.sub_(self.mean).div_(self.std)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        idx = self.next_idx
        if input is not None:
            input.record_stream(torch.cuda.current_stream())
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        if idx is not None:
            idx.record_stream(torch.cuda.current_stream())

        self.preload()
        return input, target, idx


class data_prefetcher2():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        # self.mean = torch.tensor([0.485, 0.456, 0.406]).cuda().view(1, 3, 1, 1)
        self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1, 3, 1, 1)
        # self.std = torch.tensor([0.229, 0.224, 0.225]).cuda().view(1, 3, 1, 1)
        self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1, 3, 1, 1)
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_input1, self.next_target, self.next_idx = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_input1 = None
            self.next_target = None
            self.next_idx = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_input1 = self.next_input1.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            self.next_idx = self.next_idx.cuda(non_blocking=True)
            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            self.next_input = self.next_input.float()
            self.next_input = self.next_input.sub_(self.mean).div_(self.std)

            self.next_input1 = self.next_input1.float()
            self.next_input1 = self.next_input1.sub_(self.mean).div_(self.std)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        input1 = self.next_input1
        target = self.next_target
        idx = self.next_idx
        if input is not None:
            input.record_stream(torch.cuda.current_stream())
        if input1 is not None:
            input1.record_stream(torch.cuda.current_stream())
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        if idx is not None:
            idx.record_stream(torch.cuda.current_stream())

        self.preload()
        return input, input1, target, idx


def train(model, model_backup, train_loader, unlabeled_eval_loader, start_epoch, args):
    print ('Start Neighborhood Contrastive Learning:')
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = BCE()
    mse = nn.MSELoss()

    spacing_loss_start_epoch = 5
    enable_spacing_loss = False
    n_clusters = 30
    beta = 5.0
    cm = CentroidManager(512, n_clusters)

    for epoch in range(start_epoch, args.epochs):

        prefetcher = data_prefetcher2(train_loader)
        num_iter = len(train_loader)

        if epoch == spacing_loss_start_epoch:
            # Extract features
            model.eval()
            all_features = []
            # for data, _, _, _ in prefetcher:
            for batch_idx in tqdm(range(num_iter)):
                data, _, _, _ = prefetcher.next()
                data = data.to(device)
                _, feat, _, _ = model(data, 'feat_logit')
                all_features.append(feat.detach().cpu().numpy())
            all_features = np.vstack(all_features)
            # Initialize
            cm.init_clusters(all_features)
            enable_spacing_loss = True
            print('Initialized spacing loss.')

        loss_record = AverageMeter()
        model.train()

        exp_lr_scheduler.step()
        w = args.rampup_coefficient * ramps.sigmoid_rampup(epoch, args.rampup_length)
        
        prefetcher = data_prefetcher2(train_loader)
        # for batch_idx, ((x, x_bar),  label, idx) in enumerate(tqdm(train_loader)):
        for batch_idx in tqdm(range(num_iter)):
            x, x_bar, label, idx = prefetcher.next()
            x, x_bar, label = x.to(device), x_bar.to(device), label.to(device)

            feat, feat_q, output1, output2 = model(x, 'feat_logit')
            feat_old, _, _, _ = model_backup(x, 'feat_logit')
            feat_bar, feat_k, output1_bar, output2_bar = model(x_bar, 'feat_logit')

            prob1, prob1_bar, prob2, prob2_bar = F.softmax(output1, dim=1), F.softmax(output1_bar, dim=1), F.softmax(
                output2, dim=1), F.softmax(output2_bar, dim=1)

            mask_lb = idx < train_loader.labeled_length

            # rank_feat = (feat[~mask_lb]).detach()
            rank_feat = (feat_old[~mask_lb]).detach()


            if args.bce_type == 'cos':
                # default: cosine similarity with threshold
                feat_row, feat_col = PairEnum(F.normalize(rank_feat, dim=1))
                tmp_distance_ori = torch.bmm(feat_row.view(feat_row.size(0), 1, -1), feat_col.view(feat_row.size(0), -1, 1))
                tmp_distance_ori = tmp_distance_ori.squeeze()
                target_ulb = torch.zeros_like(tmp_distance_ori).float() - 1
                target_ulb[tmp_distance_ori > args.costhre] = 1
            elif args.bce_type == 'RK':
                # top-k rank statics
                rank_idx = torch.argsort(rank_feat, dim=1, descending=True)
                rank_idx1, rank_idx2 = PairEnum(rank_idx)
                rank_idx1, rank_idx2 = rank_idx1[:, :args.topk], rank_idx2[:, :args.topk]
                rank_idx1, _ = torch.sort(rank_idx1, dim=1)
                rank_idx2, _ = torch.sort(rank_idx2, dim=1)
                rank_diff = rank_idx1 - rank_idx2
                rank_diff = torch.sum(torch.abs(rank_diff), dim=1)
                target_ulb = torch.ones_like(rank_diff).float().to(device)
                target_ulb[rank_diff > 0] = -1

            prob1_ulb, _ = PairEnum(prob2[~mask_lb])
            _, prob2_ulb = PairEnum(prob2_bar[~mask_lb])

            # basic loss
            # loss_ce = criterion1(output1[mask_lb], label[mask_lb])
            loss_bce = criterion2(prob1_ulb, prob2_ulb, target_ulb)
            consistency_loss = F.mse_loss(prob1, prob1_bar) + F.mse_loss(prob2, prob2_bar)
            loss = loss_bce + w * consistency_loss
            # loss = loss_ce + loss_bce + w * consistency_loss

            # Spacing loss
            if enable_spacing_loss and epoch >= spacing_loss_start_epoch:
                spacing_loss = torch.tensor(0.).to(device)
                features = feat_q.detach().cpu().numpy()
                # Do re-assigment
                cluster_ids = cm.update_assingment(features)
                # Update centroids
                elem_count = np.bincount(cluster_ids, minlength=n_clusters)
                for k in range(n_clusters):
                    if elem_count[k] == 0:
                        continue
                    cm.update_cluster(features[cluster_ids == k], k)
                # Compute loss
                batch_size = feat_q.size()[0]
                centroids = torch.FloatTensor(cm.centroids).to(device)
                for i in range(batch_size):
                    # diff = feat_q[i] - centroids[cluster_ids[i]]
                    # distance = torch.matmul(diff.view(1, -1), diff.view(-1, 1))
                    # spacing_loss += 0.5 * beta * torch.squeeze(distance)
                    spacing_loss += 0.5 * beta * mse(feat_q[i], centroids[cluster_ids[i]])
                loss += spacing_loss

            # NCL loss for unlabeled data
            loss_ncl_ulb = ncl_ulb(feat_q[~mask_lb], feat_k[~mask_lb], label[~mask_lb], epoch, False, ncl_la.memory.clone().detach())
            if epoch > 0:
                loss += loss_ncl_ulb * args.w_ncl_ulb

            # # NCL loss for labeled data
            # loss_ncl_la = ncl_la(feat_q[mask_lb], feat_k[mask_lb], label[mask_lb], epoch, True)

            # if epoch > 0:
            #     loss += loss_ncl_ulb * args.w_ncl_ulb + loss_ncl_la * args.w_ncl_la
            # else:
            #     loss += loss_ncl_la * args.w_ncl_la

            # ===================backward=====================
            loss_record.update(loss.item(), x.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        torch.save({'optimizer': optimizer.state_dict(), 'model': model.state_dict(),
                    'epoch': epoch, 'memory': ncl_ulb.state_dict(), 'memory_la': ncl_la.state_dict()}, args.model_dir[:-4] + '_inter.pth')
        args.head = 'head2'
        print('Learning rate: {}'.format(exp_lr_scheduler.get_last_lr()))
        print('Train Epoch: {} Avg Loss: {:.4f}'.format(epoch, loss_record.avg))
        print('test on unlabeled classes')
        test(model, unlabeled_eval_loader, args)


def test(model, test_loader, args):
    n_classes = 30

    model.eval() 
    preds=np.array([])
    targets=np.array([])
    features = []

    prefetcher = data_prefetcher(test_loader)
    x, label, idx = prefetcher.next()
    num_iter = len(test_loader)
    for i in tqdm(range(num_iter)):
    # for batch_idx, (x, label, idx) in enumerate(tqdm(test_loader)):
        x, label = x.to(device), label.to(device)
        _, feat_q, output1, output2 = model(x, 'feat_logit')
        if args.head=='head1':
            output = output1
        else:
            output = output2
        _, pred = output.max(1)
        targets=np.append(targets, label.cpu().numpy())
        preds=np.append(preds, pred.cpu().numpy())
        features.extend(feat_q.detach().cpu().numpy())

        x, label, idx = prefetcher.next()
    
    acc, nmi, ari = cluster_acc(targets.astype(int), preds.astype(int)), nmi_score(targets, preds), ari_score(targets, preds)
    print('From logits \t: Test acc {:.4f}, nmi {:.4f}, ari {:.4f}'.format(acc, nmi, ari))

    if args.kmeans_type == 'gpu_euclid':
        predictions, _ = KMeans_GPU(feat_q)

    predictions = KMeans(n_clusters=n_classes, n_init=20).fit_predict(np.array(features))
    acc_f, nmi_f, ari_f = cluster_acc(targets.astype(int), predictions.astype(int)), nmi_score(targets, predictions), ari_score(targets, predictions)
    print('From features\t: Test acc {:.4f}, nmi {:.4f}, ari {:.4f}'.format(acc_f, nmi_f, ari_f))


def copy_param(model, pretrain_dir):
    pre_dict = torch.load(pretrain_dir)
    new=list(pre_dict.items())
    dict_len = len(pre_dict.items())
    model_kvpair=model.state_dict()
    count=0
    for key, value in model_kvpair.items():
        if count < dict_len:
            layer_name,weights=new[count]      
            model_kvpair[key]=weights
            count+=1
        else:
            break
    model.load_state_dict(model_kvpair)
    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
            description='cluster',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--device_ids', default=[0], type=int, nargs='+',
                            help='device ids assignment (e.g 0 1 2 3)')
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--rampup_length', default=50, type=int)
    parser.add_argument('--rampup_coefficient', type=float, default=10.0)
    parser.add_argument('--step_size', default=30, type=int)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--unlabeled_batch_size', default=128, type=int)
    parser.add_argument('--num_labeled_classes', default=882, type=int)
    parser.add_argument('--num_unlabeled_classes', default=30, type=int)
    parser.add_argument('--dataset_root', type=str, default='./data/datasets/ImageNet/')
    parser.add_argument('--exp_root', type=str, default='./data/experiments/')
    parser.add_argument('--warmup_model_dir', type=str, default='./data/experiments/pretrained/resnet18_imagenet_classif_882_ICLR18.pth')
    parser.add_argument('--topk', default=5, type=int)
    parser.add_argument('--model_name', type=str, default='resnet_imagenet_882_pretrained_plus_50_rampupcoeff_50_old_model')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--unlabeled_subset', type=str, default='A')
    parser.add_argument('--w_ncl_la', type=float, default=0.1)
    parser.add_argument('--w_ncl_ulb', type=float, default=1.0)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--bce_type', type=str, default='cos')
    parser.add_argument('--hard_negative_start', default=3000, type=int)
    parser.add_argument('--knn', default=-1, type=int)
    parser.add_argument('--costhre', type=float, default=0.95)
    parser.add_argument('--m_size', default=3000, type=int)
    parser.add_argument('--m_t', type=float, default=0.05)
    parser.add_argument('--w_pos', type=float, default=0.2)
    parser.add_argument('--hard_iter', type=int, default=5)
    parser.add_argument('--num_hard', type=int, default=400)
    parser.add_argument('--fast_dataloader', type=bool, default=True)
    parser.add_argument('--spacing_loss_start_epoch', type=int, default=5)
    parser.add_argument('--kmeans_type', type=str, default='cpu') # cpu, gpu_euclid, gpu_cosine


    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    seed_torch(args.seed)
    runner_name = os.path.basename(__file__).split(".")[0]
    model_dir = os.path.join(args.exp_root, runner_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    args.model_dir = model_dir+'/'+'{}_{}.pth'.format(args.model_name, args.unlabeled_subset) 

    model = ResNet(BasicBlock, [2,2,2,2], args.num_labeled_classes, args.num_unlabeled_classes)
    model = nn.DataParallel(model, args.device_ids).to(device)
    model = copy_param(model, args.warmup_model_dir)

    # Backup model for generating the pseudo-labels
    model_backup = ResNet(BasicBlock, [2,2,2,2], args.num_labeled_classes, args.num_unlabeled_classes)
    model_backup = nn.DataParallel(model_backup, args.device_ids).to(device)
    model_backup = copy_param(model_backup, args.warmup_model_dir)

    num_classes = args.num_labeled_classes + args.num_unlabeled_classes

    for name, param in model.named_parameters(): 
        if 'head' not in name and 'layer4' not in name:
            param.requires_grad = False

    for name, param in model_backup.named_parameters(): 
        param.requires_grad = False

    num_workers = 12
    if args.fast_dataloader:
        # use fast data loader
        # mix_train_loader = ImageNetLoader882_30Mix_pre(args.batch_size, num_workers=8, path=args.dataset_root, unlabeled_subset=args.unlabeled_subset, aug='twice_pre', shuffle=True, subfolder='train', unlabeled_batch_size=args.unlabeled_batch_size)
        # (used-for-base-learning) mix_train_loader = ImageNetLoader882_pre(args.batch_size, num_workers=num_workers, path=args.dataset_root, aug='twice_pre', shuffle=True, subfolder='train')
        mix_train_loader = ImageNetLoader30_pre(args.batch_size, num_workers=num_workers, path=args.dataset_root, aug='twice_pre', shuffle=True, subfolder='train', subset=args.unlabeled_subset)
        # labeled_eval_loader = ImageNetLoader882_pre(args.batch_size, num_workers=num_workers, path=args.dataset_root, aug='none_pre', shuffle=False, subfolder='val')
        unlabeled_eval_loader = ImageNetLoader30_pre(args.batch_size, num_workers=num_workers, path=args.dataset_root, subset=args.unlabeled_subset, aug='none_pre', shuffle=False, subfolder='train')
    else:
        # use slow data loader
        # mix_train_loader = ImageNetLoader882_30Mix(args.batch_size, num_workers=8, path=args.dataset_root,
        #                                                unlabeled_subset=args.unlabeled_subset, aug='twice',
        #                                                shuffle=True, subfolder='train',
        #                                                unlabeled_batch_size=args.unlabeled_batch_size)
        # (used-for-base-learning) mix_train_loader = ImageNetLoader882(args.batch_size, num_workers=num_workers, path=args.dataset_root, aug='twice_pre', shuffle=True, subfolder='train')
        mix_train_loader = ImageNetLoader30(args.batch_size, num_workers=num_workers, path=args.dataset_root, aug='twice_pre', shuffle=True, subfolder='train', subset=args.unlabeled_subset)
        # labeled_eval_loader = ImageNetLoader882(args.batch_size, num_workers=num_workers, path=args.dataset_root,
        #                                             aug='none', shuffle=False, subfolder='val')
        unlabeled_eval_loader = ImageNetLoader30(args.batch_size, num_workers=num_workers, path=args.dataset_root,
                                                     subset=args.unlabeled_subset, aug='none', shuffle=False,
                                                     subfolder='train')

    ncl_ulb = NCLMemory(512, 3000, args.m_t, args.num_unlabeled_classes, args.knn, args.w_pos, args.hard_iter, args.num_hard, args.hard_negative_start).to(device)
    ncl_la = NCLMemory(512, 30000, args.m_t, args.num_labeled_classes, args.knn, args.w_pos, args.hard_iter, args.num_hard, args.hard_negative_start).to(device)

    if args.resume:
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        ncl_ulb.load_state_dict(checkpoint['memory'])
        ncl_la.load_state_dict(checkpoint['memory_la'])
        print('Start from Epoch:{}'.format(start_epoch))
    else:
        start_epoch = 0

    if args.mode == 'train':
        train(model, model_backup, mix_train_loader, unlabeled_eval_loader, start_epoch, args)
        torch.save(model.state_dict(), args.model_dir)
        print("model saved to {}.".format(args.model_dir))
    else:
        print("model loaded from {}.".format(args.model_dir))
        model.load_state_dict(torch.load(args.model_dir))

    # print('test on labeled classes')
    # args.head = 'head1'
    # test(model, labeled_eval_loader, args)
    print('test on unlabeled classes')
    args.head = 'head2'
    test(model, unlabeled_eval_loader, args)

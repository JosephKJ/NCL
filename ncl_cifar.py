import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, lr_scheduler
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.cluster import KMeans, DBSCAN
from utils.util import BCE, PairEnum, cluster_acc, Identity, AverageMeter, seed_torch, BCE_softlabels
from utils import ramps 
from models.resnet import ResNet, BasicBlock
from data.cifarloader import CIFAR10Loader, CIFAR10LoaderMix, CIFAR100Loader, CIFAR100LoaderMix
from data.svhnloader import SVHNLoader, SVHNLoaderMix
from tqdm import tqdm
import numpy as np
import os
from models.NCL import NCLMemory
from utils.spacing import CentroidManager


def train(model, train_loader, unlabeled_eval_loader, args):
    print ('Start Neighborhood Contrastive Learning:')
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = BCE()
    mse = nn.MSELoss()

    spacing_loss_start_epoch = 150
    enable_spacing_loss = False
    n_clusters = 100
    beta = 0.005
    cm = CentroidManager(512, n_clusters)

    for epoch in range(args.epochs):

        if epoch == spacing_loss_start_epoch:
            # Extract features
            model.eval()
            all_features = []
            for (data, _), _, _ in train_loader:
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
        for batch_idx, ((x, x_bar),  label, idx) in enumerate(tqdm(train_loader)):

            x, x_bar, label = x.to(device), x_bar.to(device), label.to(device)
            idx = idx.to(device)

            mask_lb = label < args.num_labeled_classes

            feat, feat_q, output1, output2 = model(x, 'feat_logit')
            feat_bar, feat_k, output1_bar, output2_bar = model(x_bar, 'feat_logit')

            prob1, prob1_bar, prob2, prob2_bar = F.softmax(output1, dim=1), F.softmax(output1_bar, dim=1), F.softmax(
                output2, dim=1), F.softmax(output2_bar, dim=1)

            rank_feat = (feat[~mask_lb]).detach()
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
            loss_ce = criterion1(output1[mask_lb], label[mask_lb])
            loss_bce = criterion2(prob1_ulb, prob2_ulb, target_ulb)
            consistency_loss = F.mse_loss(prob1, prob1_bar) + F.mse_loss(prob2, prob2_bar)
            loss = loss_ce + loss_bce + w * consistency_loss

            # Spacing loss
            if enable_spacing_loss:
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

            # NCL loss for labeled data
            loss_ncl_la = ncl_la(feat_q[mask_lb], feat_k[mask_lb], label[mask_lb], epoch, True)

            if epoch > 0:
                loss += loss_ncl_ulb * args.w_ncl_ulb + loss_ncl_la * args.w_ncl_la
            else:
                loss += loss_ncl_la * args.w_ncl_la

            # ===================backward=====================
            loss_record.update(loss.item(), x.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        args.head = 'head2'
        print('Train Epoch: {} Avg Loss: {:.4f}'.format(epoch, loss_record.avg))
        print('test on unlabeled classes')
        test(model, unlabeled_eval_loader, args)


def train_old(model, train_loader, unlabeled_eval_loader, args):
    print ('Start Neighborhood Contrastive Learning:')
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = BCE()

    for epoch in range(args.epochs):
        loss_record = AverageMeter()
        model.train()

        exp_lr_scheduler.step()
        w = args.rampup_coefficient * ramps.sigmoid_rampup(epoch, args.rampup_length)
        for batch_idx, ((x, x_bar),  label, idx) in enumerate(tqdm(train_loader)):

            x, x_bar, label = x.to(device), x_bar.to(device), label.to(device)
            idx = idx.to(device)

            mask_lb = label < args.num_labeled_classes

            feat, feat_q, output1, output2 = model(x, 'feat_logit')
            feat_bar, feat_k, output1_bar, output2_bar = model(x_bar, 'feat_logit')

            prob1, prob1_bar, prob2, prob2_bar = F.softmax(output1, dim=1), F.softmax(output1_bar, dim=1), F.softmax(
                output2, dim=1), F.softmax(output2_bar, dim=1)

            rank_feat = (feat[~mask_lb]).detach()
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
            loss_ce = criterion1(output1[mask_lb], label[mask_lb])
            loss_bce = criterion2(prob1_ulb, prob2_ulb, target_ulb)
            consistency_loss = F.mse_loss(prob1, prob1_bar) + F.mse_loss(prob2, prob2_bar)
            loss = loss_ce + loss_bce + w * consistency_loss

            # # NCL loss for unlabeled data
            # loss_ncl_ulb = ncl_ulb(feat_q[~mask_lb], feat_k[~mask_lb], label[~mask_lb], epoch, False, ncl_la.memory.clone().detach())

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

        args.head = 'head2'
        print('Train Epoch: {} Avg Loss: {:.4f}'.format(epoch, loss_record.avg))
        print('test on unlabeled classes')
        test(model, unlabeled_eval_loader, args)


def test(model, test_loader, args):
    model.eval()
    preds = np.array([])
    targets = np.array([])
    features = []
    for batch_idx, (x, label, _) in enumerate(tqdm(test_loader)):
        x, label = x.to(device), label.to(device)
        feat, feat_q, output1, output2 = model(x, 'feat_logit')
        if args.head == 'head1':
            output = output1
        else:
            output = output2
        _, pred = output.max(1)
        targets = np.append(targets, label.cpu().numpy())
        preds = np.append(preds, pred.cpu().numpy())
        features.extend(feat_q.detach().cpu().numpy())

    predictions = KMeans(n_clusters=20, n_init=20).fit_predict(np.array(features))
    acc, nmi, ari = cluster_acc(targets.astype(int), preds.astype(int)), nmi_score(targets, preds), ari_score(targets, preds)
    acc_f, nmi_f, ari_f = cluster_acc(targets.astype(int), predictions.astype(int)), nmi_score(targets, predictions), ari_score(targets, predictions)
    print('From logits \t: Test acc {:.4f}, nmi {:.4f}, ari {:.4f}'.format(acc, nmi, ari))
    print('From features\t: Test acc {:.4f}, nmi {:.4f}, ari {:.4f}'.format(acc_f, nmi_f, ari_f))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
            description='cluster',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--rampup_length', default=150, type=int)
    parser.add_argument('--rampup_coefficient', type=float, default=50)
    parser.add_argument('--step_size', default=170, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_unlabeled_classes', default=5, type=int)
    parser.add_argument('--num_labeled_classes', default=5, type=int)
    parser.add_argument('--dataset_root', type=str, default='./data/datasets/CIFAR/')
    parser.add_argument('--exp_root', type=str, default='./data/experiments/')
    parser.add_argument('--warmup_model_dir', type=str, default='./data/experiments/pretrained/auto_novel/resnet_rotnet_cifar10.pth')
    parser.add_argument('--topk', default=5, type=int)
    parser.add_argument('--model_name', type=str, default='resnet')
    parser.add_argument('--dataset_name', type=str, default='cifar10', help='options: cifar10, cifar100')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--bce_type', type=str, default='cos')
    parser.add_argument('--hard_negative_start', default=1000, type=int)
    parser.add_argument('--knn', default=-1, type=int)
    parser.add_argument('--w_ncl_la', type=float, default=0.1)
    parser.add_argument('--w_ncl_ulb', type=float, default=1.0)
    parser.add_argument('--costhre', type=float, default=0.95)
    parser.add_argument('--m_size', default=2000, type=int)
    parser.add_argument('--m_t', type=float, default=0.05)
    parser.add_argument('--w_pos', type=float, default=0.2)
    parser.add_argument('--hard_iter', type=int, default=5)
    parser.add_argument('--num_hard', type=int, default=400)

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    torch.backends.cudnn.benchmark = True
    seed_torch(args.seed)
    runner_name = os.path.basename(__file__).split(".")[0]
    model_dir = os.path.join(args.exp_root, runner_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    args.model_dir = model_dir+'/'+'{}.pth'.format(args.model_name) 

    model = ResNet(BasicBlock, [2,2,2,2], args.num_labeled_classes, args.num_unlabeled_classes).to(device)

    num_classes = args.num_labeled_classes + args.num_unlabeled_classes

    def copy_param(model, pretrain_dir):
        pre_dict = torch.load(pretrain_dir)
        new = list(pre_dict.items())
        dict_len = len(pre_dict.items())
        model_kvpair = model.state_dict()
        count = 0
        for count in range(dict_len):
            layer_name, weights = new[count]
            if 'contrastive_head' not in layer_name and 'shortcut' not in layer_name:
                if 'backbone' in layer_name:
                    model_kvpair[layer_name[9:]] = weights
                # else:
                #     model_kvpair[layer_name] = weights
                print (layer_name[9:])
            else:
                continue
        model.load_state_dict(model_kvpair)
        return model

    if args.mode == 'train':
        state_dict = torch.load(args.warmup_model_dir)
        model.load_state_dict(state_dict, strict=False)

        for name, param in model.named_parameters():
            if 'head' not in name and 'layer4' not in name:
                param.requires_grad = False

    if args.dataset_name == 'cifar10':
        mix_train_loader = CIFAR10LoaderMix(root=args.dataset_root, batch_size=args.batch_size, split='train', aug='twice', shuffle=True, labeled_list=range(args.num_labeled_classes), unlabeled_list=range(args.num_labeled_classes, num_classes))
        unlabeled_eval_loader = CIFAR10Loader(root=args.dataset_root, batch_size=args.batch_size, split='train', aug=None, shuffle=False, target_list = range(args.num_labeled_classes, num_classes))
        unlabeled_eval_loader_test = CIFAR10Loader(root=args.dataset_root, batch_size=args.batch_size, split='test', aug=None, shuffle=False, target_list = range(args.num_labeled_classes, num_classes))
        labeled_eval_loader = CIFAR10Loader(root=args.dataset_root, batch_size=args.batch_size, split='test', aug=None, shuffle=False, target_list = range(args.num_labeled_classes))
    elif args.dataset_name == 'cifar100':
        mix_train_loader = CIFAR100LoaderMix(root=args.dataset_root, batch_size=args.batch_size, split='train', aug='twice', shuffle=True, labeled_list=range(args.num_labeled_classes), unlabeled_list=range(args.num_labeled_classes, num_classes))
        unlabeled_eval_loader = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='train', aug=None, shuffle=False, target_list = range(args.num_labeled_classes, num_classes))
        unlabeled_eval_loader_test = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='test', aug=None, shuffle=False, target_list = range(args.num_labeled_classes, num_classes))
        labeled_eval_loader = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='test', aug=None, shuffle=False, target_list = range(args.num_labeled_classes))

    ncl_ulb = NCLMemory(512, args.m_size, args.m_t, args.num_unlabeled_classes, args.knn, args.w_pos, args.hard_iter, args.num_hard, args.hard_negative_start).to(device)
    ncl_la = NCLMemory(512, args.m_size, args.m_t, args.num_labeled_classes, args.knn, args.w_pos, args.hard_iter, args.num_hard, args.hard_negative_start).to(device)

    if args.mode == 'train':
        train(model, mix_train_loader, unlabeled_eval_loader, args)
        torch.save(model.state_dict(), args.model_dir)
        print("model saved to {}.".format(args.model_dir))
    else:
        print("model loaded from {}.".format(args.model_dir))
        model.load_state_dict(torch.load(args.model_dir))

    print('Evaluating on Head1')
    args.head = 'head1'
    print('test on labeled classes (test split)')
    test(model, labeled_eval_loader, args)

    print('Evaluating on Head2')
    args.head = 'head2'
    print('test on unlabeled classes (train split)')
    test(model, unlabeled_eval_loader, args)
    print('test on unlabeled classes (test split)')
    test(model, unlabeled_eval_loader_test, args)
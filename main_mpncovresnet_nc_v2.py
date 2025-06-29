from __future__ import print_function

import argparse
import math
import os
import random

from sklearn.metrics import roc_curve

import sys
import time
from confusion_pytorch import EntropicConfusion, PairwiseConfusion
import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import pickle

import pprint
from torchvision import transforms
from dataset import CANDatasetEnet as CANDataset
from losses import SupConLoss, cdef_loss, fdef_loss_v2
from util import AverageMeter, AddGaussianNoise, TwoCropTransform
from rd_openmax import fit_weibull_rd, openmax_predict_rd, extract_class_stats 
from util import warmup_learning_rate
from util import save_model, accuracy
from networks.con_mpncovresnet import ConTinyMPNCOVResNet , LinearClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, accuracy_score
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=100,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=1,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')
    parser.add_argument('--resume', type=str, default=None, 
                        help='path to the checkpoint to resume from')
    
    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.03,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    # optimization classifier
    parser.add_argument('--epoch_start_classifier', type=int, default=90)
    parser.add_argument('--learning_rate_classifier', type=float, default=0.01,
                        help='learning rate classifier')
    parser.add_argument('--lr_decay_epochs_classifier', type=str, default='60,75,90',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate_classifier', type=float, default=0.2,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay_classifier', type=float, default=0,
                        help='weight decay_classifier')
    parser.add_argument('--momentum_classifier', type=float, default=0.9,
                        help='momentum_classifier')

    # model dataset
    parser.add_argument('--model', type=str, default='efficient-net',)
    parser.add_argument('--optimizer', type=str, default='SGD')
    parser.add_argument('--dataset', type=str, default='CAN_TT',
                        choices=['CAN-ML', 'CAN-TT'],
                        help='dataset')
    parser.add_argument('--mean', type=str, 
                        help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, 
                        help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str, default=None, 
                        help='path to custom dataset')
    parser.add_argument('--close_set_test_data', type=str, default=None, 
                        help='path to custom dataset')
    parser.add_argument('--open_set_test_data', type=str, default=None,
                    help='Path to additional test data (for RD-OpenMax)')
    parser.add_argument('--n_classes', type=int, default=6, 
                        help='number of class')

    # openmax
    parser.add_argument('--test_contains_unknown', action='store_true',
                        help='Use RD-OpenMax to detect unknown classes during test')
    parser.add_argument('--known_class_label', type=int, default=0)
    parser.add_argument('--representation', type=str, default=None,
                        choices=['GAvP', 'MPNCOV', 'BCNN', 'CBP'],
                        help='optional representation layer after backbone (e.g., GAvP, MPNCOV)')
    parser.add_argument('--attention', type=str, default=None,
                        help='Attention module name')
    parser.add_argument('--loss', type=str, default='p')
    # method
    parser.add_argument('--method', type=str, default='UniCon', 
                        choices=['UniCon', 'SupCon', 'SimCLR'],
                        help='choose method')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')
    opt = parser.parse_args()

    opt.device = torch.device('cuda:1')

    opt.model_path = './save/{}_models/{}'.format(opt.dataset, opt.method)
    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_{}_lr_{}_decay_{}_bsz_{}_trial_{}'. \
        format(opt.method, opt.dataset, opt.model, opt.learning_rate,
               opt.weight_decay, opt.batch_size, opt.trial)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)
    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)
    # dir to save log file
    opt.log_file = f'{opt.model_path}/{opt.model_name}/log'
    opt.tb_folder = f'{opt.model_path}/{opt.model_name}/runs'
    return opt

def get_predict(outputs):
    _, pred = outputs.topk(1, 1, True, True)
    pred = pred.t().cpu().numpy().squeeze(0)
    return pred

def set_loader(opt):
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)
    normalize = transforms.Normalize(mean=mean, std=std)
    transform = transforms.Compose([normalize])
    train_transform = transforms.Compose([
        transforms.RandomApply([AddGaussianNoise(0., 0.05)], p=0.5),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),
        transforms.Normalize(mean=mean, std=std)
    ])
    # Khởi tạo dataset
    train_dataset = CANDataset(root_dir=opt.data_folder, window_size=32, is_train=True, transform=TwoCropTransform(train_transform))
    close_set_test_dataset = CANDataset(root_dir=opt.close_set_test_data, window_size=32, is_train=False, transform=transform)

    open_set_test_loader = None
    if opt.open_set_test_data:
        open_set_test_dataset = CANDataset(root_dir=opt.open_set_test_data, window_size=32, is_train=False, transform=transform)
        open_set_test_loader = torch.utils.data.DataLoader(open_set_test_dataset, batch_size=opt.batch_size, shuffle=False,  pin_memory=True)

    train_classifier_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, 
        shuffle=True, num_workers=opt.num_workers,
        pin_memory=True, sampler=None)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers, pin_memory=True, drop_last=True)
    close_set_test_loader = torch.utils.data.DataLoader(close_set_test_dataset, batch_size=opt.batch_size, shuffle=False, pin_memory=True)

    return train_loader, train_classifier_loader, close_set_test_loader, open_set_test_loader


def set_model(opt):
    model = ConTinyMPNCOVResNet(
        attention=opt.attention,
        input_size=64,
        feat_dim=128
    )
    criterion = SupConLoss(temperature=0.07)
    criterion_classifier = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    classifier = LinearClassifier(input_dim=8256, num_classes=opt.n_classes)
        
    model = model.to(opt.device)
    criterion = criterion.to(opt.device)
    classifier = classifier.to(opt.device)
    criterion_classifier = criterion_classifier.to(opt.device)

    cudnn.benchmark = True

    print('Model device: ', next(model.parameters()).device)
    return model, criterion, classifier, criterion_classifier

optimize_dict = {
    'SGD' : optim.SGD,
    'RMSprop': optim.RMSprop,
    'Adam': optim.Adam,
    'AdamW': optim.AdamW
}

def set_optimizer(opt, model, class_str='', optim_choice='SGD'):
    dict_opt = vars(opt)
    optimizer = optimize_dict[optim_choice]

    # Handle both Adam and AdamW cases
    if optim_choice in ['Adam', 'AdamW']:
        optimizer = optimizer(
            model.parameters(),
            lr=dict_opt['learning_rate' + class_str],
            weight_decay=dict_opt['weight_decay' + class_str]
        )
    else:
        # For optimizers that require momentum like SGD and RMSprop
        optimizer = optimizer(
            model.parameters(),
            lr=dict_opt['learning_rate' + class_str],
            momentum=dict_opt['momentum' + class_str],
            weight_decay=dict_opt['weight_decay' + class_str]
        )

    return optimizer

def train(train_loader, model, criterion, optimizer, epoch, opt, step):
    """Stage 1: Training encoder with SupCon + F-DEF loss"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()

    for idx, (images, labels) in enumerate(train_loader):
        step += 1
        data_time.update(time.time() - end)

        image1, image2 = images[0], images[1]
        images = torch.cat([image1, image2], dim=0)

        images = images.to(opt.device, non_blocking=True)
        labels = labels.to(opt.device, non_blocking=True)
        bsz = labels.shape[0]

        # Warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # --- Forward ---
        features = model.encoder(images)
        features = F.normalize(features, dim=1)

        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        supcon_features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        # --- SupCon loss ---
        loss_supcon = criterion(supcon_features, labels)

        # --- F-DEF loss ---
        all_features = torch.cat([f1, f2], dim=0)
        all_labels = torch.cat([labels, labels], dim=0)

        loss_fdef = fdef_loss_v2(all_features, all_labels)
        loss_fdef = torch.clamp(loss_fdef, max=10.0)
        # loss_fdef = fdef_loss(f1, labels)
        # Combine
        alpha = 1.0  # weight for SupCon, tune if needed
        beta = 1.5   # weight for F-DEF  #old => 1.5
        loss = alpha * loss_supcon + beta * loss_fdef

        # Update
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"NaN detected! SupCon: {loss_supcon.item()}, FDEF: {loss_fdef.item()}")
            continue  # skip batch

        losses.update(loss.item(), bsz)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if (idx + 1) % opt.print_freq == 0:
            log_message = (
                'Train: [{0}][{1}/{2}]\t'
                'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'loss {loss.val:.3f} ({loss.avg:.3f})\n'.format(
                    epoch, idx + 1, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses
                )
            )
            print(log_message)
            sys.stdout.flush()

    return step, losses.avg

def train_classifier(train_loader, model, classifier, optimizer, epoch, opt, step, logger, scale_pos=1.0, scale_neg=1.0):
    model.eval()
    classifier.train()

    losses = AverageMeter()
    accs = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        step += 1
        images = images[0]
        images = images.to(opt.device, non_blocking=True)
        labels = labels.to(opt.device, non_blocking=True)
        bsz = labels.size(0)

        # === 1. Extract features from the encoder ===
        with torch.no_grad():
            features = model.encoder(images)

        # === 2. Predict logits with classifier ===
        logits = classifier(features.detach())

        # === 3. Compute C-DEF Loss ===
        ce_loss = F.cross_entropy(logits, labels)
        cdef = cdef_loss(logits, labels, margin=0.5)
        loss = ce_loss + 0.1 * cdef

        losses.update(loss.item(), bsz)

        # === 4. Accuracy for monitoring ===
        acc = accuracy(logits, labels, topk=(1,))
        accs.update(acc[0].item(), bsz)

        # === 5. Backward and optimize ===
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # === 6. Logging ===
        if step % opt.print_freq == 0:
            logger.add_scalar('loss/train', losses.avg, step)
            logger.add_scalar('accuracy/train', accs.avg, step)

    return step, losses.avg, accs.avg



def validate_closed_set(val_loader, model, classifier, criterion, opt):
    model.eval()
    classifier.eval()

    losses = AverageMeter()
    total_pred = []
    total_label = []

    with torch.no_grad():
        progress_bar = tqdm(val_loader, file=sys.stdout, disable=not sys.stdout.isatty())
        for images, labels in progress_bar:
            if isinstance(images, (list, tuple)):
                images = images[0]

            images = images.to(opt.device, non_blocking=True)
            labels = labels.to(opt.device, non_blocking=True)

            features = model.encoder(images)         # 👉 Extract features
            outputs = classifier(features)           # 👉 Classify

            loss = criterion(outputs, labels)
            losses.update(loss.item(), labels.size(0))

            preds = get_predict(outputs)
            total_pred.extend(preds)
            total_label.extend(labels.cpu().numpy())

    acc = accuracy_score(total_label, total_pred) * 100
    f1 = f1_score(total_label, total_pred, average='weighted')
    precision = precision_score(total_label, total_pred, average='weighted', zero_division=0)
    recall = recall_score(total_label, total_pred, average='weighted')
    conf_matrix = confusion_matrix(total_label, total_pred, labels=list(range(opt.n_classes)))

    sys.stdout.flush()
    return losses.avg, acc, f1, precision, recall, conf_matrix

def find_best_threshold(y_true, unk_probs):
    # --- Check input validity ---
    y_true = np.array(y_true)
    unk_probs = np.array(unk_probs)

    if np.isnan(unk_probs).any():
        print("Warning: NaN detected in unk_probs, replacing with 0.0")
        unk_probs = np.nan_to_num(unk_probs, nan=0.0)

    if np.isinf(unk_probs).any():
        print("Warning: Inf detected in unk_probs, clipping to finite range")
        unk_probs = np.clip(unk_probs, 0.0, 1.0)

    # --- Compute ROC and Youden's J-statistic ---
    fpr, tpr, thresholds = roc_curve(y_true, unk_probs)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    
    return thresholds[best_idx], fpr[best_idx], tpr[best_idx]

def validate_open_set(val_loader, model, opt, class_means, weibull_models, classifier=None):
    model.eval()
    if classifier is not None:
        classifier.eval()

    total_preds = []
    total_labels = []
    unk_probs = []

    with torch.no_grad():
        progress_bar = tqdm(val_loader, file=sys.stdout, disable=not sys.stdout.isatty())
        for images, labels in progress_bar:
            if isinstance(images, (list, tuple)):
                images = images[0]
            images = images.to(opt.device, non_blocking=True)
            labels = labels.to(opt.device, non_blocking=True)

            # ----- Lấy feature embedding -----
            if classifier is not None:
                feats = model.encoder(images)  # [B, feat_dim]
            else:
                feats = model(images)          # [B, feat_dim]

            feats = F.normalize(feats, dim=1)
            feats_np = feats.cpu().numpy()

            # ----- Dự đoán với OpenMax -----
            for f, label in zip(feats_np, labels.cpu().numpy()):
                pred_class, unk_prob = openmax_predict_rd(f, class_means, weibull_models)
                total_preds.append((pred_class, unk_prob))
                total_labels.append(label)
                unk_probs.append(unk_prob)

    # Mapping: 0 -> known, 1 -> unknown
    known_label = getattr(opt, 'known_class_label', 0)
    mapped_label = [0 if l == known_label else 1 for l in total_labels]

    # Tìm ngưỡng tối ưu
    best_thresh, best_fpr, best_tpr = find_best_threshold(mapped_label, unk_probs)
    print(f'>>> Best Threshold = {best_thresh:.4f} | FPR: {best_fpr:.4f} | TPR: {best_tpr:.4f}')

    # Phân loại theo ngưỡng
    mapped_pred = [0 if p[1] < best_thresh else 1 for p in total_preds]

    # Tính chỉ số hiệu suất
    acc = accuracy_score(mapped_label, mapped_pred) * 100
    f1 = f1_score(mapped_label, mapped_pred, average='binary')
    precision = precision_score(mapped_label, mapped_pred, average='binary', zero_division=0)
    recall = recall_score(mapped_label, mapped_pred, average='binary')
    conf_matrix = confusion_matrix(mapped_label, mapped_pred)

    return 0, acc, f1, precision, recall, conf_matrix


def adjust_learning_rate(args, optimizer, epoch, class_str=''):
    dict_args = vars(args)
    lr = dict_args['learning_rate'+class_str]
    if args.cosine:
        eta_min = lr * (dict_args['lr_decay_rate'+class_str] ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        lr_decay_epochs_arr = list(map(int, dict_args['lr_decay_epochs'+class_str].split(',')))
        lr_decay_epochs_arr = np.asarray(lr_decay_epochs_arr)
        steps = np.sum(epoch > lr_decay_epochs_arr)
        if steps > 0:
            lr = lr * (dict_args['lr_decay_rate'+class_str] ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main():
    opt = parse_option()
    print(opt)

    seed = 1
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # === Data Loader ===
    train_loader, train_classifier_loader, close_set_test_loader, open_set_test_loader = set_loader(opt)

    # === Model ===
    model, criterion, classifier, criterion_classifier = set_model(opt)

    # === Optimizer ===
    optimizer = set_optimizer(opt, model, optim_choice=opt.optimizer)
    optimizer_classifier = set_optimizer(opt, classifier, class_str='_classifier', optim_choice=opt.optimizer)

    logger = SummaryWriter(log_dir=opt.tb_folder, flush_secs=2)
    log_writer = open(opt.log_file, 'w')

    start_epoch = 1
    step = 0

    for epoch in range(start_epoch, opt.epochs + 1):
        print(f'=== Epoch {epoch} | Time: {time.strftime("%Y-%m-%d %H:%M:%S")} ===')
        adjust_learning_rate(opt, optimizer, epoch)

        # === Training ===
        step, train_loss = train(train_loader, model, criterion, optimizer, epoch, opt, step)
        print(f'Epoch {epoch}, Train SupCon Loss: {train_loss:.4f}')
        sys.stdout.flush()
        log_writer.write(f'Epoch: {epoch}, Train SupCon Loss: {train_loss:.4f}\n')
        
        # === Validation ===
        if epoch % opt.save_freq == 0:
            pp = pprint.PrettyPrinter(indent=4)
            print("------------Train classifier...------------")
            adjust_learning_rate(opt, optimizer_classifier, epoch, '_classifier')
            new_step, loss_ce, train_acc = train_classifier(train_classifier_loader, model, classifier, 
                                                            optimizer_classifier, epoch, opt, step, logger)
            print(f'Epoch {epoch}, Train Classifier Loss: {loss_ce:.4f} | Train Acc: {train_acc:.2f}')

            print("------------Validating on closed-set...------------")
            sys.stdout.flush()
            val_loss_close, val_acc_close, val_f1_close, precision_close, recall_close, conf_matrix_close = validate_closed_set(
                close_set_test_loader, model, classifier, criterion_classifier, opt
            )
            print("Closed-set Results:")
            print(f'Acc: {val_acc_close:.2f} | F1: {val_f1_close:.4f} | Precision: {precision_close:.4f} | Recall: {recall_close:.4f}')
            print('Confusion Matrix:')
            print(conf_matrix_close)
            sys.stdout.flush()
            if opt.test_contains_unknown:
                print("------------Fitting Weibull models...------------")
                sys.stdout.flush()
                class_feats, class_means = extract_class_stats(model, classifier, train_loader, opt.device)
                weibull_models = fit_weibull_rd(class_feats, class_means, tailsize=50)

                with open(os.path.join(opt.save_folder, f'weibull_epoch_{epoch}.pkl'), 'wb') as f:
                    pickle.dump({
                        'class_means': class_means,
                        'weibull_models': weibull_models
                    }, f)
                print("------------Validating with RD-OpenMax...------------")
                val_loss_open, val_acc_open, val_f1_open, precision_open, recall_open, conf_matrix_open = validate_open_set(
                    open_set_test_loader, model, opt, class_means, weibull_models, classifier
                )
                print(f'Open-set Results:\nAcc: {val_acc_open:.2f} | F1: {val_f1_open:.4f} | '
                    f'Precision: {precision_open:.4f} | Recall: {recall_open:.4f}')
                print('Confusion Matrix:')
                print(conf_matrix_open)
                sys.stdout.flush()

        # === Save Checkpoint ===
        if epoch % opt.save_freq == 0:
            save_path = os.path.join(opt.save_folder, f'ckpt_epoch_{epoch}.pth')
            save_model(model, optimizer, opt, epoch, save_path)

            save_path_classifier = os.path.join(opt.save_folder, f'ckpt_classifier_epoch_{epoch}.pth')
            save_model(classifier, optimizer_classifier, opt, epoch, save_path_classifier)  

    # === Save Final Model ===
    save_path = os.path.join(opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_path)

    
if __name__ == '__main__':
    main()
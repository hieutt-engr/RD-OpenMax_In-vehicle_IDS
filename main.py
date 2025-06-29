from __future__ import print_function

import argparse
import math
import os
import random
import sys
import time
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
from losses import SupConLoss, UniConLoss
from networks.efficient_net_v2 import ConEfficientNetV2, LinearClassifier
from util import TwoCropTransform, AverageMeter, AddGaussianNoise
from rd_openmax import fit_weibull_rd, openmax_predict_rd, extract_class_stats
from util import warmup_learning_rate
from util import get_universum
from util import save_model ,load_checkpoint, accuracy
# from networks.classifier import LinearClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, accuracy_score
from torch.utils.tensorboard import SummaryWriter
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
    # mixup
    parser.add_argument('--lamda', type=float, default=0.5, 
                        help='universum lambda')
    parser.add_argument('--mix', type=str, default='mixup', 
                        choices=['mixup', 'cutmix'], 
                        help='use mixup or cutmix')
    parser.add_argument('--size', type=int, default=32, 
                        help='parameter for RandomResizedCrop')

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

    # method
    parser.add_argument('--method', type=str, default='UniCon', 
                        choices=['UniCon', 'SupCon', 'SimCLR'],
                        help='choose method')

    # temperature
    parser.add_argument('--temp', type=float, default=0.07, 
                        help='temperature for loss function')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')
    opt = parser.parse_args()

    opt.device = torch.device('cuda:0')

    opt.model_path = './save/{}_models/{}'.format(opt.dataset, opt.method)
    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_{}_lr_{}_decay_{}_bsz_{}_temp_{}_{}_lambda_{}_trial_{}'. \
        format(opt.method, opt.dataset, opt.model, opt.learning_rate,
               opt.weight_decay, opt.batch_size, opt.temp, opt.mix, opt.lamda, opt.trial)

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

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers, pin_memory=True)
    train_classifier_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, 
        shuffle=True, num_workers=opt.num_workers,
        pin_memory=True, sampler=None)
    close_set_test_loader = torch.utils.data.DataLoader(close_set_test_dataset, batch_size=opt.batch_size, shuffle=False, pin_memory=True)

    return train_loader, train_classifier_loader, close_set_test_loader, open_set_test_loader


def set_model(opt):
    model = ConEfficientNetV2(embedding_dim=1280, feat_dim=128, head='mlp', pretrained=False)
    if opt.method == 'UniCon':
        criterion = UniConLoss(temperature=opt.temp)
    else:
        criterion = SupConLoss(temperature=opt.temp)

    classifier = LinearClassifier(input_dim=1280, num_classes=opt.n_classes)
    criterion_classifier = torch.nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder, device_ids=[0])
        
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
    """one epoch training"""
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

        universum = get_universum(images, labels, opt)
        uni_features = model(universum)

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        features = model(images)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

        loss = criterion(features, uni_features, labels)

        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # print info
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

def train_classifier(train_loader, model, classifier, criterion, optimizer, epoch, opt, step, logger):
    model.eval()
    classifier.train()

    losses = AverageMeter()
    accs = AverageMeter()

    # Start the training loop
    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        step += 1
        images = images[0]
        # if torch.cuda.is_available():
        #     images = images.cuda(non_blocking=True)
        #     labels = labels.cuda(non_blocking=True)
        images = images.to(opt.device, non_blocking=True)
        labels = labels.to(opt.device, non_blocking=True)
        bsz = labels.size(0)  # Batch size

        # Extract features from the pre-trained model in evaluation mode
        with torch.no_grad():
            features = model.encoder(images)

        # Forward pass through the classifier
        output = classifier(features.detach())

        # Compute loss
        loss = criterion(output, labels)
        losses.update(loss.item(), bsz)

        # Compute accuracy
        acc = accuracy(output, labels, topk=(1,))
        accs.update(acc[0].item(), bsz)

        # Backward pass and optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log loss and accuracy to TensorBoard every opt.print_freq steps
        if step % opt.print_freq == 0:
            logger.add_scalar('loss/train', losses.avg, step)
            logger.add_scalar('accuracy/train', accs.avg, step)

    # Return the updated step count, average loss, and average accuracy
    return step, losses.avg, accs.avg

def get_predict(outputs):
    _, pred = outputs.topk(1, 1, True, True)
    pred = pred.t().cpu().numpy().squeeze(0)
    return pred


def validate_close_set(val_loader, model, classifier, criterion, opt):
    model.eval()
    classifier.eval()
    
    losses = AverageMeter()
    total_pred = np.array([], dtype=int)
    total_label = np.array([], dtype=int) 
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader): 
            # if torch.cuda.is_available():
            #     images = images.cuda(non_blocking=True)
            #     labels = labels.cuda(non_blocking=True)
            images = images.to(opt.device, non_blocking=True)
            labels = labels.to(opt.device, non_blocking=True)
            features = model.encoder(images)
            outputs = classifier(features)

            bsz = labels.size(0)
            loss = criterion(outputs, labels)
            losses.update(loss.item(), bsz)
            
            pred = get_predict(outputs)

            if isinstance(pred, torch.Tensor):
                pred = pred.cpu().numpy()
            
            if isinstance(labels, torch.Tensor):
                labels = labels.cpu().numpy()

            total_pred = np.concatenate((total_pred, pred), axis=0)
            total_label = np.concatenate((total_label, labels), axis=0)
    
    acc = accuracy_score(total_label, total_pred)
    acc = acc * 100
    f1 = f1_score(total_label, total_pred, average='weighted')
    precision = precision_score(total_label, total_pred, average='weighted', zero_division=0)
    recall = recall_score(total_label, total_pred, average='weighted')
    conf_matrix = confusion_matrix(total_label, total_pred)
    
    return losses.avg, acc, f1, precision, recall, conf_matrix

def validate_open_set(val_loader, model, classifier, opt, class_means, weibull_models):
    model.eval()
    if classifier is not None:
        classifier.eval()

    total_pred = []
    total_label = []

    with torch.no_grad():
        for images, labels in val_loader:
            if isinstance(images, (list, tuple)):
                images = images[0]
            images = images.to(opt.device)
            labels = labels.to(opt.device)

            if classifier is not None:
                feats = model.encoder(images)        # [B, 1280]
                feats = classifier(feats)            # [B, num_classes]
            else:
                feats = model(images)                # [B, feat_dim or logits]
            
            feats = feats.cpu().numpy()
            preds = [openmax_predict_rd(f, class_means, weibull_models) for f in feats]

            total_pred.extend(preds)
            total_label.extend(labels.cpu().numpy())

    known_label = getattr(opt, 'known_class_label', 0)
    mapped_label = [0 if l == known_label else 1 for l in total_label]
    mapped_pred = [0 if p != -1 else 1 for p in total_pred]

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

    # build data loader
    train_loader, train_classifier_loader, close_set_test_loader, open_set_test_loader = set_loader(opt)
    model, criterion, classifier, criterion_classifier = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model, optim_choice=opt.optimizer)
    optimizer_classifier = set_optimizer(opt, classifier, class_str='_classifier', optim_choice=opt.optimizer)
    
    logger = SummaryWriter(log_dir=opt.tb_folder, flush_secs=2)

    start_epoch = 1  # Default start from epoch 1
    step = 0

    # Loading from a checkpoint
    if opt.resume:
        new_epoch = opt.epochs
        new_save_freq = opt.save_freq
        checkpoint_path = opt.model_path + '/' + opt.model_name + '/' + opt.resume
        model, optimizer, start_epoch, opt = load_checkpoint(checkpoint_path, model, optimizer)
        if new_epoch != opt.epochs:
            opt.epochs = new_epoch
        if new_save_freq != opt.save_freq:
            opt.save_freq = new_save_freq
        print(f"Resuming training from epoch {start_epoch} to epoch {opt.epochs}...")

    log_writer = open(opt.log_file, 'w')
    # training routine
    for epoch in range(start_epoch, opt.epochs + 1):
        print('Begin time: ' + str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())))
        adjust_learning_rate(opt, optimizer, epoch)

        new_step, train_loss = train(train_loader, model, criterion, optimizer, epoch, opt, step)

        print(f'Epoch {epoch}, Unicon Loss {train_loss:.4f}')
        log_writer.write(f'Epoch: {epoch}, Unicon Loss: {train_loss:.4f}\n')

        # Train and validate classifier 
        # if epoch > opt.epoch_start_classifier:
        if epoch % opt.save_freq == 0:
            print("Train Classifier...")
            adjust_learning_rate(opt, optimizer_classifier, epoch, '_classifier')
            new_step, loss_ce, train_acc = train_classifier(train_classifier_loader, model, classifier, 
                                                            criterion_classifier, optimizer_classifier, epoch, opt, step, logger)
            pp = pprint.PrettyPrinter(indent=4)
            log_writer.write('Classifier: Loss: {:.4f}, Acc: {}\n'.format(loss_ce, train_acc))

            print("Validation Classifier...")
            loss_close, val_acc_close, val_f1_close, precision_close, recall_close, conf_matrix_close = validate_close_set(close_set_test_loader, model, classifier, criterion_classifier, opt)
            print("Closed-set Results:")
            print(f'Acc: {val_acc_close:.2f} | F1: {val_f1_close:.4f} | Precision: {precision_close:.4f} | Recall: {recall_close:.4f}')
            print('Confusion Matrix:')
            print(conf_matrix_close)
            # Tensorboard logging
            logger.add_scalar('loss_ce/train', loss_ce, step)
            logger.add_scalar('loss_ce/val', loss_close, step)
            logger.add_scalar('f1/val', val_f1_close, step)
            logger.add_scalar('precision/val', precision_close, step)
            logger.add_scalar('recall/val', recall_close, step)

            log_writer.write(f'F1 Score: {val_f1_close:.4f}\n')
            log_writer.write(f'Precision: {precision_close:.4f}\n')
            log_writer.write(f'Recall: {recall_close:.4f}\n')
            log_writer.write(f'Confusion Matrix:\n{pp.pformat(conf_matrix_close)}\n')

            if opt.test_contains_unknown: 
                print("------------Fitting Weibull models...------------")
                sys.stdout.flush()
                # class_feats, class_means = extract_class_stats(model, None, train_loader, opt.device)
                # weibull_models = fit_weibull_rd(class_feats, class_means)

                class_feats, class_means = extract_class_stats(model, classifier, train_loader, opt.device)
                weibull_models = fit_weibull_rd(class_feats, class_means, tailsize=20)

                with open(os.path.join(opt.save_folder, f'weibull_epoch_{epoch}.pkl'), 'wb') as f:
                    pickle.dump({
                        'class_means': class_means,
                        'weibull_models': weibull_models
                    }, f)

                print("------------Validating with RD-OpenMax...------------")
                val_loss_open, val_acc_open, val_f1_open, precision_open, recall_open, conf_matrix_open = validate_open_set(
                    open_set_test_loader, model, classifier, opt, class_means, weibull_models
                )

                print(f'Open-set Results:\nAcc: {val_acc_open:.2f} | F1: {val_f1_open:.4f} | '
                    f'Precision: {precision_open:.4f} | Recall: {recall_open:.4f}')
                print('Confusion Matrix:')
                print(conf_matrix_open)
                sys.stdout.flush()

        step = new_step
        # Save checkpoint 
        if epoch % opt.save_freq == 0:
            save_file = os.path.join(opt.save_folder, f'ckpt_epoch_{epoch}.pth')
            save_model(model, optimizer, opt, epoch, save_file)

            ckpt = 'ckpt_class_epoch_{}.pth'.format(epoch)
            save_file = os.path.join(opt.save_folder, ckpt)
            save_model(classifier, optimizer_classifier, opt, epoch, save_file)
    
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)

    save_file = os.path.join(
        opt.save_folder, 'last_classifier.pth')
    save_model(classifier, optimizer_classifier, opt, opt.epochs, save_file)
    
if __name__ == '__main__':
    main()
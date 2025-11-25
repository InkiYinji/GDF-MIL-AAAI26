import os
import torch
import time
import random
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from datasets.dataloader import DatasetWSI, DatasetBag
from datasets.MIL import get_k_cv_idx
from models.DGRMIL.dgr_utils import tripleloss
from utils.basic import get_optimizer, get_dtfd_optimizer, get_criterion, get_scheduler, cal_scores


def get_model(args):
    if args.general.model_name == "GDAMIL":
        from models.GDAMIL.GDAMIL import GDAMIL
        mil_model = GDAMIL(args.in_dim, args.general.num_classes,
                              dropout=args.model.dropout,
                              k_components=args.model.k_components,
                              k_neighbors=args.model.k_neighbors
                              )
    elif args.general.model_name == "GDAMIL1":
        from models.GDAMIL1.GDAMIL import GDAMIL
        mil_model = GDAMIL(args.in_dim, args.general.num_classes,
                              dropout=args.model.dropout,
                              k_components=args.model.k_components,
                              k_neighbors=args.model.k_neighbors
                              )
    else:
        raise Exception("No such model!")

    return mil_model


def get_dataset_info_feat_path(args, dataset):
    if args.feature_extractor == "resnet18":
        train_dataset_info_feat_path = dataset.data_save_home + dataset.save.resnet18.training.h5
        test_dataset_info_feat_path = dataset.data_save_home + dataset.save.resnet18.testing.h5
    elif args.feature_extractor == "camil":
        train_dataset_info_feat_path = dataset.data_save_home + dataset.save.camil.training.h5
        test_dataset_info_feat_path = dataset.data_save_home + dataset.save.camil.testing.h5
    elif args.feature_extractor == "dsmil":
        train_dataset_info_feat_path = dataset.data_save_home + dataset.save.dsmil.training.h5
        test_dataset_info_feat_path = dataset.data_save_home + dataset.save.dsmil.testing.h5
    elif args.feature_extractor == "clip_vitb32":
        train_dataset_info_feat_path = dataset.data_save_home + dataset.save.clip_vitb32.training.h5
        test_dataset_info_feat_path = dataset.data_save_home + dataset.save.clip_vitb32.testing.h5
    elif args.feature_extractor == "conch":
        train_dataset_info_feat_path = dataset.data_save_home + dataset.save.conch.training.h5
        test_dataset_info_feat_path = dataset.data_save_home + dataset.save.conch.testing.h5
    else:
        train_dataset_info_feat_path = dataset.data_save_home + dataset.save.resnet50.training.h5
        test_dataset_info_feat_path = dataset.data_save_home + dataset.save.resnet50.testing.h5
    return train_dataset_info_feat_path, test_dataset_info_feat_path


def train_kcv(args, mil, record_writer, record_writer_best):
    """
    General settings.
    """
    record_writer.write2file("Set seed: %s" % args.general.seed)

    device = torch.device(f'cuda:{args.general.device}') if torch.cuda.is_available() else torch.device("cpu")
    tr_idx, te_idx = get_k_cv_idx(mil.N, k=args.k, seed=args.general.seed)
    for fold in range(args.k):
        record_writer_best.write2file("Fold: %d" % fold)
        record_writer.write2file("Fold: %d" % fold)
        record_writer.write2file("Load dataset...")

        train_dataloader = DatasetBag(mil.bag_space, mil.bag_lab, tr_idx[fold])
        test_dataloader = DatasetBag(mil.bag_space, mil.bag_lab, te_idx[fold])

        record_writer.write2file("DataLoader ready!")
        record_writer.write2file("Get model...")

        if args.general.model_name == "DKMIL":
            from models.DKMIL.knowledge import DataDriven
            from models.DKMIL.DKMIL import DKMIL
            knowledge = DataDriven(args, args.data_path)
            knowledge.fit(tr_idx[fold])
            mil_model = DKMIL(args, num_classes=args.general.num_classes, knowledge=knowledge).to(device)
        else:
            mil_model = get_model(args)

        if args.general.model_name in ["DGRMIL", "GKSMIL"]:
            mil_model = mil_model.to(device)
        else:
            mil_model = nn.DataParallel(mil_model).to(device)
        optimizer, base_lr = get_optimizer(args, mil_model)
        scheduler, warmup_scheduler = get_scheduler(args, optimizer, base_lr)
        criterion = get_criterion(args.model.criterion)
        record_writer.write2file("Model ready!")

        """
        Training
        """
        record_writer.write2file("----------------INFO----------------")
        record_writer.write2file("Start training...")
        best_score = -1

        model_save_path = os.path.join(args.log_dir, args.experiments_type
                                       + "_seed" + str(args.general.seed) + "_fold" + str(fold) + ".pth")
        record_writer.write2file("Model save path: %s" % model_save_path)
        if os.path.exists(model_save_path) and args.retrain is False:
            record_writer.write2file("Load pretrained model...")
            mil_model.load_state_dict(torch.load(str(model_save_path)))
            _, best_scores = test_epoch(args, mil_model, test_dataloader, criterion, args.general.num_classes,
                                        1000, device, record_writer)
            output_all_predicted_scores(record_writer_best, best_scores, -1, -1, -1, -1)
        else:
            for epoch in range(args.general.num_epochs):

                record_writer.write2file("\tEpoch %d:" % epoch, is_print=False)

                if epoch+1 <= args.model.scheduler.warmup:
                    now_scheduler = warmup_scheduler
                else:
                    now_scheduler = scheduler
                train_loss, cost_time = train_epoch(args, mil_model, train_dataloader, criterion,
                                                    optimizer, now_scheduler, epoch, device)
                test_loss, predicted_scores = test_epoch(args, mil_model, test_dataloader, criterion,
                                                         args.general.num_classes, epoch, device, record_writer)
                output_all_predicted_scores(record_writer, predicted_scores, epoch, train_loss, test_loss, cost_time)

                if predicted_scores[args.general.best_metric] > best_score:
                    best_epoch = epoch + 1
                    best_score = predicted_scores[args.general.best_metric]
                    torch.save(mil_model.state_dict(), str(model_save_path))
                    record_writer_best.write2file("Best epoch: %d" % best_epoch, is_print=False)
                    output_all_predicted_scores(record_writer_best, predicted_scores, epoch, train_loss, test_loss,
                                                cost_time, is_print=False)


def train_dtfd(args, dataset, record_writer, record_writer_best):
    """
    General settings.
    """
    record_writer.write2file("Set seed: %s" % args.general.seed)
    generator = seed_setup(args.general.seed)

    record_writer.write2file("Load dataset...")
    train_path = dataset.data_save_home + dataset.save.patch.training.process_list
    test_path = dataset.data_save_home + dataset.save.patch.testing.process_list

    train_dataset_info_feat_path, test_dataset_info_feat_path = get_dataset_info_feat_path(args, dataset)

    train_dataset = DatasetWSI(train_path, train_dataset_info_feat_path, data_type=dataset.data_type)
    test_dataset = DatasetWSI(test_path, test_dataset_info_feat_path, data_type=dataset.data_type)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, generator=generator)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    record_writer.write2file("DataLoader ready!")
    record_writer.write2file("Get model...")
    device = torch.device(f'cuda:{args.general.device}') if torch.cuda.is_available() else torch.device("cpu")

    from models.DTFDMIL import Classifier_1fc, Attention, DimReduction, Attention_with_Classifier
    classifier = Classifier_1fc(args.model.mdim, args.general.num_classes, args.model.classifier_dropout).to(device)
    attention = Attention(args.model.mdim).to(device)
    dimReduction = DimReduction(args.in_dim, args.model.mdim, numLayer_Res=args.model.numLayer_Res).to(device)
    attCls = Attention_with_Classifier(L=args.model.mdim, num_cls=args.general.num_classes,
                                       droprate=args.model.attCls_dropout).to(device)
    mil_model = [classifier, attention, dimReduction, attCls]

    trainable_parameters_A = []
    trainable_parameters_A.extend(list(classifier.parameters()))
    trainable_parameters_A.extend(list(attention.parameters()))
    trainable_parameters_A.extend(list(dimReduction.parameters()))
    trainable_parameters_B = attCls.parameters()

    optimizer_A, base_lr = get_dtfd_optimizer(args, trainable_parameters_A)
    scheduler_A, warmup_scheduler_A = get_scheduler(args, optimizer_A, base_lr)

    optimizer_B, base_lr = get_dtfd_optimizer(args, trainable_parameters_B)
    scheduler_B, warmup_scheduler_B = get_scheduler(args, optimizer_B, base_lr)

    optimizer = [optimizer_A, optimizer_B]

    criterion = get_criterion(args.model.criterion)
    record_writer.write2file("Model ready!")

    """
    Training
    """
    record_writer.write2file("----------------INFO----------------")
    record_writer.write2file("Start training...")
    best_score = -1

    for epoch in range(args.general.num_epochs):

        record_writer.write2file("\tEpoch %d:" % epoch, is_print=False)

        if epoch+1 <= args.model.scheduler.warmup:
            now_scheduler_A = warmup_scheduler_A
            now_scheduler_B = warmup_scheduler_B
        else:
            now_scheduler_A = scheduler_A
            now_scheduler_B = scheduler_B
        now_scheduler = [now_scheduler_A, now_scheduler_B]
        train_loss, cost_time = train_epoch_dtfd(args, mil_model, train_dataloader, criterion,
                                                 optimizer, now_scheduler, device)
        test_loss, predicted_scores = test_epoch_dtfd(args, mil_model, test_dataloader, criterion,
                                                      args.general.num_classes, epoch, device, record_writer)

        output_all_predicted_scores(record_writer, predicted_scores, epoch, train_loss, test_loss, cost_time)

        if predicted_scores[args.general.best_metric] > best_score:
            best_epoch = epoch + 1
            best_score = predicted_scores[args.general.best_metric]
            record_writer_best.write2file("Best epoch: %d" % best_epoch, is_print=False)
            output_all_predicted_scores(record_writer_best, predicted_scores, epoch, train_loss, test_loss,
                                        cost_time, is_print=False)


def train_kcv_tdfd(args, mil, record_writer, record_writer_best):
    """
    General settings.
    """
    record_writer.write2file("Set seed: %s" % args.general.seed)

    device = torch.device(f'cuda:{args.general.device}') if torch.cuda.is_available() else torch.device("cpu")
    tr_idx, te_idx = get_k_cv_idx(mil.N, k=args.k, seed=args.general.seed)
    for fold in range(args.k):

        record_writer_best.write2file("Fold: %d" % fold)
        record_writer.write2file("Fold: %d" % fold)
        record_writer.write2file("Load dataset...")

        train_dataloader = DatasetBag(mil.bag_space, mil.bag_lab, tr_idx[fold])
        test_dataloader = DatasetBag(mil.bag_space, mil.bag_lab, te_idx[fold])

        record_writer.write2file("DataLoader ready!")
        record_writer.write2file("Get model...")

        from models.DTFDMIL import Classifier_1fc, Attention, DimReduction, Attention_with_Classifier
        classifier = Classifier_1fc(args.model.mdim, args.general.num_classes, args.model.classifier_dropout).to(device)
        attention = Attention(args.model.mdim).to(device)
        dimReduction = DimReduction(args.in_dim, args.model.mdim, numLayer_Res=args.model.numLayer_Res).to(device)
        attCls = Attention_with_Classifier(L=args.model.mdim, num_cls=args.general.num_classes,
                                           droprate=args.model.attCls_dropout).to(device)
        mil_model = [classifier, attention, dimReduction, attCls]

        trainable_parameters_A = []
        trainable_parameters_A.extend(list(classifier.parameters()))
        trainable_parameters_A.extend(list(attention.parameters()))
        trainable_parameters_A.extend(list(dimReduction.parameters()))
        trainable_parameters_B = attCls.parameters()

        optimizer_A, base_lr = get_dtfd_optimizer(args, trainable_parameters_A)
        scheduler_A, warmup_scheduler_A = get_scheduler(args, optimizer_A, base_lr)

        optimizer_B, base_lr = get_dtfd_optimizer(args, trainable_parameters_B)
        scheduler_B, warmup_scheduler_B = get_scheduler(args, optimizer_B, base_lr)

        optimizer = [optimizer_A, optimizer_B]

        criterion = get_criterion(args.model.criterion)
        record_writer.write2file("Model ready!")

        """
        Training
        """
        record_writer.write2file("----------------INFO----------------")
        record_writer.write2file("Start training...")
        best_score = -1

        for epoch in range(args.general.num_epochs):

            # record_writer.write2file("\tEpoch %d:" % epoch, is_print=False)

            if epoch + 1 <= args.model.scheduler.warmup:
                now_scheduler_A = warmup_scheduler_A
                now_scheduler_B = warmup_scheduler_B
            else:
                now_scheduler_A = scheduler_A
                now_scheduler_B = scheduler_B
            now_scheduler = [now_scheduler_A, now_scheduler_B]
            train_loss, cost_time = train_epoch_dtfd(args, mil_model, train_dataloader, criterion,
                                                     optimizer, now_scheduler, device)
            test_loss, predicted_scores = test_epoch_dtfd(args, mil_model, test_dataloader, criterion,
                                                          args.general.num_classes, epoch, device, record_writer)
            output_all_predicted_scores(record_writer, predicted_scores, epoch, train_loss, test_loss, cost_time)

            if predicted_scores[args.general.best_metric] > best_score:
                best_epoch = epoch + 1
                best_score = predicted_scores[args.general.best_metric]
                record_writer_best.write2file("Best epoch: %d" % best_epoch, is_print=False)
                output_all_predicted_scores(record_writer_best, predicted_scores, epoch, train_loss, test_loss,
                                            cost_time, is_print=False)


def output_all_predicted_scores(record_writer, predicted_scores, epoch, train_loss, test_loss, cost_time, is_print=True):
    record_writer.write2file(
        "\tEPOCH: %d,  train loss: %.4f, test loss: %.4f, time: %.4f, "
        "acc: %.4f, f1_score: %.4f, auc: %.4f"
        % (epoch + 1, train_loss, test_loss, cost_time,
           predicted_scores["acc"], predicted_scores["f1"], predicted_scores["auc"],
           ), is_print=is_print)
    # record_writer.write2file(
    #     "\tEPOCH: %d,  train loss: %.4f, test loss: %.4f, time: %.4f, "
    #     "acc: %.4f, bacc: %.4f, "
    #     "macro_auc: %.4f, micro_auc: %.4f, weighted_auc: %.4f, "
    #     "macro_f1: %.4f, micro_f1: %.4f, weighted_f1: %.4f, "
    #     "macro_recall: %.4f, micro_recall: %.4f, weighted_recall: %.4f, "
    #     "macro_pre: %.4f, micro_pre: %.4f, weighted_pre: %.4f"
    #     % (epoch + 1, train_loss, test_loss, cost_time,
    #        predicted_scores["acc"], predicted_scores["bacc"],
    #        predicted_scores["macro_auc"], predicted_scores["micro_auc"], predicted_scores["weighted_auc"],
    #        predicted_scores["macro_f1"], predicted_scores["micro_f1"], predicted_scores["weighted_f1"],
    #        predicted_scores["macro_recall"], predicted_scores["micro_recall"], predicted_scores["weighted_recall"],
    #        predicted_scores["macro_pre"], predicted_scores["micro_pre"], predicted_scores["weighted_pre"],
    #        ), is_print=is_print)


def train_epoch(args, mil_model, train_dataloader, criterion, optimizer, scheduler, epoch, device):
    start = time.time()

    mil_model.train()
    total_train_loss = 0

    for i, data in enumerate(train_dataloader):

        bag, label = data
        bag = bag.to(device).float()
        label = label.to(device).long()

        optimizer.zero_grad()

        max_prediction = 0
        if args.general.model_name == "TestMIL":
            train_logits, A = mil_model(bag)
        elif args.general.model_name == "DSMIL":
            train_logits, max_prediction, A = mil_model(bag)
            max_prediction, _ = torch.max(max_prediction, 0)
            max_prediction = max_prediction.unsqueeze(0)
        elif args.general.model_name in ["CLAMMB", "CLAMSB"]:
            train_logits, A = mil_model(bag, label=label)
        elif args.general.model_name == "DGRMIL":
            train_logits, A, _, dgrmil_positive, dgrmil_negative, dgrmil_lesion = mil_model(bag, bag_mode='lesion')
        elif args.general.model_name == "MILGNN":
            train_logits = mil_model(bag)
        elif args.general.model_name == "GKSMIL":
            train_logits, A, _, dgrmil_positive, dgrmil_negative, dgrmil_lesion = mil_model(bag, bag_mode='negative',
                                                                                            train_phase=True)
        elif args.general.model_name == "TADGraph":
            train_logits, A, info_loss = mil_model(bag, current_epoch=epoch)
        elif args.general.model_name in ["BagGraph", "RGMIL"]:
            train_logits, A = mil_model(bag, device)
        elif args.general.model_name in ["MyGNN"]:
            train_logits, A, loss_smooth, loss_nce = mil_model(bag)
        else:
            train_logits, A = mil_model(bag)

        del bag

        train_loss = criterion(train_logits, label)

        if args.general.model_name == "DSMIL":
            loss_max = criterion(max_prediction, label)
            train_loss = 0.5 * train_loss + 0.5 * loss_max
        if args.general.model_name == "DGRMIL":
            if epoch + 1 <= args.model.scheduler.warmup:
                lesion_norm = dgrmil_lesion.squeeze(0)
                lesion_norm = torch.nn.functional.normalize(lesion_norm)
                # div_loss = -torch.logdet(lesion_norm @ lesion_norm.T + 1e-10 * torch.eye(args.model.num_les).cuda())
                sim_loss = tripleloss(dgrmil_lesion, dgrmil_positive, dgrmil_negative)
                train_loss = train_loss + 0.1 * sim_loss
                # train_loss = train_loss + 0.1 * div_loss + 0.1 * sim_loss

        if args.general.model_name == "TADGraph":
            train_loss = train_loss + 0.1 * info_loss

        if args.general.model_name == "GKSMIL":
            if epoch + 1 <= args.model.scheduler.warmup:
                sim_loss = tripleloss(dgrmil_lesion, dgrmil_positive, dgrmil_negative)
                train_loss = train_loss + 0.5 * sim_loss

        if args.general.model_name in ["MyGNN"]:
            train_loss = train_loss + loss_smooth + loss_nce

        train_loss.backward()
        optimizer.step()

        total_train_loss = total_train_loss + train_loss.detach().item()
        torch.cuda.empty_cache()

    if scheduler is not None:
        scheduler.step()

    total_train_loss /= len(train_dataloader)
    end = time.time()
    total_time = end - start
    return total_train_loss, total_time


def test_epoch(args, mil_model, test_dataloader, criterion, num_classes, epoch, device, record_writer):

    mil_model.eval()
    total_test_loss = 0
    labels = []
    predicted_labels = []
    with torch.no_grad():
        for i, data in enumerate(test_dataloader):
            bag, label = data
            bag = bag.to(device).float()
            label = label.to(device).long()
            labels.append(label.cpu().numpy())

            max_prediction = 0
            if args.general.model_name == "TestMIL":
                test_logits, A = mil_model(bag)
            elif args.general.model_name == "DSMIL":
                test_logits, max_prediction, A = mil_model(bag)
                max_prediction, _ = torch.max(max_prediction, 0)
                max_prediction = max_prediction.unsqueeze(0)
            elif args.general.model_name in ["CLAMMB", "CLAMSB"]:
                test_logits, A = mil_model(bag, label=label)
            elif args.general.model_name == "DGRMIL":
                test_logits, A, _= mil_model(bag, bag_mode='lesion')
            elif args.general.model_name == "MILGNN":
                test_logits = mil_model(bag)
            elif args.general.model_name == "GKSMIL":
                test_logits, A, _, dgrmil_positive, dgrmil_negative, dgrmil_lesion = mil_model(bag,
                                                                                                bag_mode='negative',
                                                                                                train_phase=True)
            elif args.general.model_name == "TADGraph":
                test_logits, A, info_loss = mil_model(bag, current_epoch=epoch, split="test")
            elif args.general.model_name in ["BagGraph", "RGMIL"]:
                test_logits, A = mil_model(bag, device)
            elif args.general.model_name in ["MyGNN"]:
                test_logits, A, loss_smooth, loss_nce = mil_model(bag)
            else:
                test_logits, A = mil_model(bag)

            test_logits = test_logits.detach()

            if args.general.model_name == "DSMIL":
                loss_max = criterion(max_prediction, label)
                test_logits = 0.5 * test_logits + 0.5 * loss_max

            # if args.general.model_name == "MILGNN":
            #     test_logits = test_logits + 0.1 * auxiliary_loss

            del bag

            predicted_labels.append(torch.softmax(test_logits.squeeze(0), dim=0).cpu().numpy())
            test_loss = criterion(test_logits, label)

            if args.general.model_name == "TADGraph":
                test_loss = test_loss + 0.1 * info_loss

            if args.general.model_name in ["MyGNN"]:
                test_loss = test_loss + loss_smooth + loss_nce

            total_test_loss = total_test_loss + test_loss.detach().item()

            # del bag, label
            torch.cuda.empty_cache()

    total_test_loss /= len(test_dataloader)

    test_metrics = cal_scores(predicted_labels, labels)

    predicted_labels = [np.argmax(label) for label in predicted_labels]
    labels = [label[0] for label in labels]
    if epoch == 0:
        record_writer.write2file("Predicted labels: %s" % predicted_labels, False)
    record_writer.write2file("True labels: %s" % labels, False)

    return total_test_loss, test_metrics


def train_epoch_dtfd(args, mil_model, train_dataloader, criterion, optimizer, scheduler, device):
    start = time.time()

    classifier, attention, dimReduction, attCls = mil_model
    classifier.train()
    attention.train()
    dimReduction.train()
    attCls.train()
    optimizer_A, optimizer_B = optimizer
    scheduler_A, scheduler_B = scheduler

    total_train_loss = 0
    total_instance = args.model.total_instance
    num_Group = args.model.num_Group
    instance_per_group = total_instance // num_Group

    for i, data in enumerate(train_dataloader):
        bag, label = data
        bag = bag.to(device).float()
        label = label.to(device).long()

        slide_sub_preds = []
        slide_sub_labels = []
        slide_pseudo_feat = []
        inputs_pseudo_bags = torch.chunk(bag.squeeze(0), num_Group, dim=0)

        for subFeat_tensor in inputs_pseudo_bags:
            slide_sub_labels.append(label)
            subFeat_tensor = subFeat_tensor.to(device)

            # Forward pass through models
            tmidFeat = dimReduction(subFeat_tensor)
            tAA = attention(tmidFeat).squeeze(0)
            tattFeats = torch.einsum('ns,n->ns', tmidFeat, tAA)  # n x fs
            tattFeat_tensor = torch.sum(tattFeats, dim=0, keepdim=True)  # 1 x fs
            tPredict = classifier(tattFeat_tensor)  # 1 x 2
            patch_pred_logits = get_cam_1d(classifier, tattFeats.unsqueeze(0)).squeeze(0)  ###  cls x n
            patch_pred_logits = torch.transpose(patch_pred_logits, 0, 1)  ## n x cls
            patch_pred_softmax = torch.softmax(patch_pred_logits, dim=1)  ## n x cls

            _, sort_idx = torch.sort(patch_pred_softmax[:, -1], descending=True)
            topk_idx_max = sort_idx[:instance_per_group].long()
            topk_idx_min = sort_idx[-instance_per_group:].long()
            topk_idx = torch.cat([topk_idx_max, topk_idx_min], dim=0)
            MaxMin_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx)
            max_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx_max)
            af_inst_feat = tattFeat_tensor

            if args.model.distill == 'MaxMinS':
                slide_pseudo_feat.append(MaxMin_inst_feat)
            elif args.model.distill == 'MaxS':
                slide_pseudo_feat.append(max_inst_feat)
            elif args.model.distill == 'AFS':
                slide_pseudo_feat.append(af_inst_feat)
            slide_sub_preds.append(tPredict)

        del bag

        slide_pseudo_feat = torch.cat(slide_pseudo_feat, dim=0)
        slide_sub_preds = torch.cat(slide_sub_preds, dim=0)  # numGroup x fs
        slide_sub_labels = torch.cat(slide_sub_labels, dim=0)  # numGroup

        train_loss_A = criterion(slide_sub_preds, slide_sub_labels)
        optimizer_A.zero_grad()
        train_loss_A.backward(retain_graph=True)
        total_train_loss += train_loss_A.item()

        torch.nn.utils.clip_grad_norm_(dimReduction.parameters(), args.model.grad_clipping)
        torch.nn.utils.clip_grad_norm_(attention.parameters(), args.model.grad_clipping)
        torch.nn.utils.clip_grad_norm_(classifier.parameters(), args.model.grad_clipping)

        gSlidePred = attCls(slide_pseudo_feat)
        train_loss_B = criterion(gSlidePred, label).mean()
        optimizer_B.zero_grad()
        train_loss_B.backward()
        total_train_loss += train_loss_B.item()

        torch.nn.utils.clip_grad_norm_(attCls.parameters(), args.model.grad_clipping)

        optimizer_A.step()
        optimizer_B.step()

        torch.cuda.empty_cache()

    if scheduler is not None:
        scheduler_A.step()
        scheduler_B.step()

    total_train_loss /= len(train_dataloader)
    end = time.time()
    total_time = end - start
    return total_train_loss, total_time


def get_cam_1d(classifier, features):
    tweight = list(classifier.parameters())[-2]
    cam_maps = torch.einsum('bgf,cf->bcg', [features, tweight])
    return cam_maps


def test_epoch_dtfd(args, mil_model, test_dataloader, criterion, num_classes, epoch, device, record_writer):

    classifier, attention, dimReduction, attCls = mil_model
    classifier.eval()
    attention.eval()
    dimReduction.eval()
    attCls.eval()

    total_test_loss = 0
    labels = []
    predicted_labels = []
    total_instance = args.model.total_instance
    num_Group = args.model.num_Group
    instance_per_group = total_instance // num_Group

    with torch.no_grad():
        for i, data in enumerate(test_dataloader):
            bag, label = data
            bag = bag.to(device).float()
            label = label.to(device).long()
            labels.append(label.cpu().numpy())

            slide_sub_preds = []
            slide_sub_labels = []
            slide_pseudo_feat = []

            inputs_pseudo_bags = torch.chunk(bag.squeeze(0), num_Group, dim=0)

            for subFeat_tensor in inputs_pseudo_bags:
                subFeat_tensor = subFeat_tensor.to(device)
                with torch.no_grad():
                    tmidFeat = dimReduction(subFeat_tensor)
                    tAA = attention(tmidFeat).squeeze(0)
                    tattFeats = torch.einsum('ns,n->ns', tmidFeat, tAA)  # n x fs
                    tattFeat_tensor = torch.sum(tattFeats, dim=0, keepdim=True)  # 1 x fs
                    tPredict = classifier(tattFeat_tensor)  # 1 x 2
                tattFeats = torch.einsum('ns,n->ns', tmidFeat, tAA)  ### n x fs
                tattFeat_tensor = torch.sum(tattFeats, dim=0).unsqueeze(0)  ## 1 x fs
                patch_pred_logits = get_cam_1d(classifier, tattFeats.unsqueeze(0)).squeeze(0)  ###  cls x n
                patch_pred_logits = torch.transpose(patch_pred_logits, 0, 1)  ## n x cls
                patch_pred_softmax = torch.softmax(patch_pred_logits, dim=1)  ## n x cls

                _, sort_idx = torch.sort(patch_pred_softmax[:, -1], descending=True)
                topk_idx_max = sort_idx[:instance_per_group].long()
                topk_idx_min = sort_idx[-instance_per_group:].long()
                topk_idx = torch.cat([topk_idx_max, topk_idx_min], dim=0)
                MaxMin_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx)
                max_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx_max)
                af_inst_feat = tattFeat_tensor

                if args.model.distill == 'MaxMinS':
                    slide_pseudo_feat.append(MaxMin_inst_feat)
                elif args.model.distill == 'MaxS':
                    slide_pseudo_feat.append(max_inst_feat)
                elif args.model.distill == 'AFS':
                    slide_pseudo_feat.append(af_inst_feat)
                slide_sub_preds.append(tPredict)

            slide_pseudo_feat = torch.cat(slide_pseudo_feat, dim=0)
            gSlidePred = torch.softmax(attCls(slide_pseudo_feat), dim=1)
            loss = criterion(gSlidePred, label)
            total_test_loss += loss.item()
            pred = (gSlidePred.cpu().data.numpy()).tolist()

            del bag

            predicted_labels.extend(pred)

            # del bag, label
            torch.cuda.empty_cache()

    total_test_loss /= len(test_dataloader)

    test__metrics = cal_scores(predicted_labels, labels)

    if epoch == 0:
        record_writer.write2file("Predicted labels: %s" % predicted_labels, False)
    record_writer.write2file("True labels: %s" % labels, False)

    return total_test_loss, test__metrics


def seed_setup(seed):
    generator = torch.Generator()
    generator.manual_seed(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False

    return generator

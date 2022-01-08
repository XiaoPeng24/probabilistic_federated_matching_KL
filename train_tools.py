import os
from utils import *
import pickle
import copy
from sklearn.preprocessing import normalize
from altmin_utils.altmin import get_mods, get_codes, update_codes, update_last_layer_, update_hidden_weights_adam_
from altmin_utils.altmin import scheduler_step, post_processing_step

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def local_retrain_fedavg(n_train, local_datasets, weights, args, device="cpu"):
    """
    freezing_index :: starting from which layer we update the model weights,
                      i.e. freezing_index = 0 means we train the whole network normally
                           freezing_index = len(model) means we freez the entire network
    """
    if args.model == "lenet":
        num_filters = [weights[0].shape[0], weights[2].shape[0]]
        kernel_size = 5
        input_dim = weights[4].shape[0]
        hidden_dims = [weights[4].shape[1]]
        output_dim = weights[-1].shape[0]
        logger.info("Num filters: {}, Input dim: {}, hidden_dims: {}, output_dim: {}".format(num_filters, input_dim, hidden_dims, output_dim))

        matched_cnn = LeNetContainer(
                                    num_filters=num_filters,
                                    kernel_size=kernel_size,
                                    input_dim=input_dim,
                                    hidden_dims=hidden_dims,
                                    output_dim=output_dim)
    elif args.model == "vgg":
        matched_shapes = [w.shape for w in weights]
        matched_cnn = matched_vgg11(matched_shapes=matched_shapes)
    elif args.model == "unet":
        matched_shapes = [w.shape for w in weights]
        matched_cnn = matched_unet(matched_shapes=matched_shapes, bilinear=args.bilinear)       
    elif args.model == "simple-cnn":
        # input_channel, num_filters, kernel_size, input_dim, hidden_dims, output_dim=10):
        # [(9, 75), (9,), (19, 225), (19,), (475, 123), (123,), (123, 87), (87,), (87, 10), (10,)]
        if args.dataset in ("cifar10", "cinic10"):
            input_channel = 3
        elif args.dataset == "mnist":
            input_channel = 1

        num_filters = [weights[0].shape[0], weights[2].shape[0]]
        input_dim = weights[4].shape[0]
        hidden_dims = [weights[4].shape[1], weights[6].shape[1]]
        matched_cnn = SimpleCNNContainer(input_channel=input_channel, 
                                        num_filters=num_filters, 
                                        kernel_size=5, 
                                        input_dim=input_dim, 
                                        hidden_dims=hidden_dims, 
                                        output_dim=10)
    elif args.model == "moderate-cnn":
        matched_cnn = ModerateCNN()

    new_state_dict = {}
    # handle the conv layers part which is not changing
    for param_idx, (key_name, param) in enumerate(matched_cnn.state_dict().items()):
        if "conv" in key_name or "features" in key_name:
            if "weight" in key_name:
                temp_dict = {key_name: torch.from_numpy(weights[param_idx].reshape(param.size()))}
            elif "bias" in key_name:
                temp_dict = {key_name: torch.from_numpy(weights[param_idx])}
        elif "fc" in key_name or "classifier" in key_name:
            if "weight" in key_name:
                temp_dict = {key_name: torch.from_numpy(weights[param_idx].T)}
            elif "bias" in key_name:
                temp_dict = {key_name: torch.from_numpy(weights[param_idx])}
        
        new_state_dict.update(temp_dict)
    matched_cnn.load_state_dict(new_state_dict)

    matched_cnn.to(device).train()
    # start training last fc layers:
    train_dl_local = local_datasets[0]
    test_dl_local = local_datasets[1]


    #optimizer_fine_tune = optim.Adam(filter(lambda p: p.requires_grad, matched_cnn.parameters()), lr=0.001, weight_decay=0.0001, amsgrad=True)
    optimizer_fine_tune = optim.SGD(filter(lambda p: p.requires_grad, matched_cnn.parameters()), lr=args.retrain_lr, momentum=0.9, weight_decay=0.0001)    
    criterion_fine_tune = nn.CrossEntropyLoss().to(device)

    logger.info('n_training: %d' % len(train_dl_local))
    logger.info('n_test: %d' % len(test_dl_local))

    train_acc = compute_accuracy(matched_cnn, train_dl_local, device=device)
    test_acc, conf_matrix = compute_accuracy(matched_cnn, test_dl_local, get_confusion_matrix=True, device=device)

    logger.info('>> Pre-Training Training accuracy: %f' % train_acc)
    logger.info('>> Pre-Training Test accuracy: %f' % test_acc)

    if args.train_online_altmin:
        # Expose model modules that has_codes
        matched_cnn = get_mods(matched_cnn, optimizer='Adam', optimizer_params={'lr': args.lr_weights},
                       scheduler=lambda epoch: 1 / 2 ** (epoch // args.lr_half_epochs))
        matched_cnn[-1].optimizer.param_groups[0]['lr'] = args.lr_out
        mu = args.mu
        mu_max = 10 * args.mu

    for epoch in range(args.retrain_epochs):
        epoch_loss_collector = []

        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.to(device), target.to(device)
            # pdb.set_trace()

            if args.train_online_altmin:
                with torch.no_grad():
                    outputs, codes = get_codes(matched_cnn, x)

                # (2) Update codes
                codes = update_codes(codes, matched_cnn, target, criterion_fine_tune, mu, lambda_c=args.lambda_c,
                                     n_iter=args.n_iter_codes, lr=args.lr_codes)

                # (3) Update weights
                update_last_layer_(matched_cnn[-1], codes[-1], target, criterion_fine_tune, n_iter=args.n_iter_weights, args, global_mods[-1])

                update_hidden_weights_adam_(matched_cnn, x, codes, lambda_w=args.lambda_w, n_iter=args.n_iter_weights, args, global_mods)

                loss = criterion_fine_tune(outputs, target)
                epoch_loss_collector.append(loss.item())

                # Increment mu
                if mu < mu_max:
                    mu = mu + args.d_mu
            else:
                optimizer_fine_tune.zero_grad()
                x.requires_grad = True
                target.requires_grad = False
                target = target.long()

                out = matched_cnn(x)
                loss = criterion_fine_tune(out, target)
                epoch_loss_collector.append(loss.item())

                loss.backward()
                optimizer_fine_tune.step()

                cnt += 1
                losses.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

    train_acc = compute_accuracy(matched_cnn, train_dl_local, device=device)
    test_acc, conf_matrix = compute_accuracy(matched_cnn, test_dl_local, get_confusion_matrix=True, device=device)

    logger.info('>> Training accuracy after local retrain: %f' % train_acc)
    logger.info('>> Test accuracy after local retrain: %f' % test_acc)
    
    return matched_cnn

def local_retrain_fedprox(n_train, local_datasets, weights, mu, args, device="cpu"):
    """
    freezing_index :: starting from which layer we update the model weights,
                      i.e. freezing_index = 0 means we train the whole network normally
                           freezing_index = len(model) means we freez the entire network
    Implementing FedProx Algorithm from: https://arxiv.org/pdf/1812.06127.pdf
    """
    if args.model == "lenet":
        num_filters = [weights[0].shape[0], weights[2].shape[0]]
        kernel_size = 5
        input_dim = weights[4].shape[0]
        hidden_dims = [weights[4].shape[1]]
        output_dim = weights[-1].shape[0]
        logger.info("Num filters: {}, Input dim: {}, hidden_dims: {}, output_dim: {}".format(num_filters, input_dim, hidden_dims, output_dim))

        matched_cnn = LeNetContainer(
                                    num_filters=num_filters,
                                    kernel_size=kernel_size,
                                    input_dim=input_dim,
                                    hidden_dims=hidden_dims,
                                    output_dim=output_dim)
    elif args.model == "vgg":
        matched_shapes = [w.shape for w in weights]
        matched_cnn = matched_vgg11(matched_shapes=matched_shapes)
    elif args.model == "unet":
        matched_shapes = [w.shape for w in weights]
        matched_cnn = matched_unet(matched_shapes=matched_shapes, bilinear=args.bilinear)
    elif args.model == "simple-cnn":
        # input_channel, num_filters, kernel_size, input_dim, hidden_dims, output_dim=10):
        # [(9, 75), (9,), (19, 225), (19,), (475, 123), (123,), (123, 87), (87,), (87, 10), (10,)]
        if args.dataset in ("cifar10", "cinic10"):
            input_channel = 3
        elif args.dataset == "mnist":
            input_channel = 1

        num_filters = [weights[0].shape[0], weights[2].shape[0]]
        input_dim = weights[4].shape[0]
        hidden_dims = [weights[4].shape[1], weights[6].shape[1]]
        matched_cnn = SimpleCNNContainer(input_channel=input_channel, 
                                        num_filters=num_filters, 
                                        kernel_size=5, 
                                        input_dim=input_dim, 
                                        hidden_dims=hidden_dims, 
                                        output_dim=10)
    elif args.model == "moderate-cnn":
        matched_cnn = ModerateCNN()

    new_state_dict = {}
    # handle the conv layers part which is not changing
    global_weight_collector = []
    for param_idx, (key_name, param) in enumerate(matched_cnn.state_dict().items()):
        if "conv" in key_name or "features" in key_name:
            if "weight" in key_name:
                temp_dict = {key_name: torch.from_numpy(weights[param_idx].reshape(param.size()))}
                global_weight_collector.append(torch.from_numpy(weights[param_idx].reshape(param.size())).to(device))
            elif "bias" in key_name:
                temp_dict = {key_name: torch.from_numpy(weights[param_idx])}
                global_weight_collector.append(torch.from_numpy(weights[param_idx]).to(device))
        elif "fc" in key_name or "classifier" in key_name:
            if "weight" in key_name:
                temp_dict = {key_name: torch.from_numpy(weights[param_idx].T)}
                global_weight_collector.append(torch.from_numpy(weights[param_idx].T).to(device))
            elif "bias" in key_name:
                temp_dict = {key_name: torch.from_numpy(weights[param_idx])}
                global_weight_collector.append(torch.from_numpy(weights[param_idx]).to(device))
        
        new_state_dict.update(temp_dict)
    matched_cnn.load_state_dict(new_state_dict)

    matched_cnn.to(device).train()
    # start training last fc layers:
    train_dl_local = local_datasets[0]
    test_dl_local = local_datasets[1]

    optimizer_fine_tune = optim.SGD(filter(lambda p: p.requires_grad, matched_cnn.parameters()), lr=args.retrain_lr, momentum=0.9, weight_decay=0.0001)
    criterion_fine_tune = nn.CrossEntropyLoss().to(device)

    logger.info('n_training: {}'.format(len(train_dl_local)))
    logger.info('n_test: {}'.format(len(test_dl_local)))

    train_acc = compute_accuracy(matched_cnn, train_dl_local, device=device)
    test_acc, conf_matrix = compute_accuracy(matched_cnn, test_dl_local, get_confusion_matrix=True, device=device)

    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

    if args.train_online_altmin:
        # Expose model modules that has_codes
        matched_cnn = get_mods(matched_cnn, optimizer='Adam', optimizer_params={'lr': args.lr_weights},
                       scheduler=lambda epoch: 1 / 2 ** (epoch // args.lr_half_epochs))
        matched_cnn[-1].optimizer.param_groups[0]['lr'] = args.lr_out
        mu = args.mu
        mu_max = 10 * args.mu

    global_mods = matched_cnn

    for epoch in range(retrain_epochs):
        epoch_loss_collector = []

        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.to(device), target.to(device)
            # pdb.set_trace()

            if args.train_online_altmin:
                with torch.no_grad():
                    outputs, codes = get_codes(matched_cnn, x)

                # (2) Update codes
                codes = update_codes(codes, matched_cnn, target, criterion_fine_tune, mu, lambda_c=args.lambda_c,
                                     n_iter=args.n_iter_codes, lr=args.lr_codes)

                # (3) Update weights
                update_last_layer_(matched_cnn[-1], codes[-1], target, criterion, n_iter=args.n_iter_weights, args,
                                   global_mods[-1])

                update_hidden_weights_adam_(matched_cnn, x, codes, lambda_w=args.lambda_w, n_iter=args.n_iter_weights, args,
                                            global_mods)

                loss = criterion_fine_tune(outputs, target)
                epoch_loss_collector.append(loss.item())

                # Increment mu
                if mu < mu_max:
                    mu = mu + args.d_mu
            else:
                optimizer_fine_tune.zero_grad()
                x.requires_grad = True
                target.requires_grad = False
                target = target.long()

                out = matched_cnn(x)
                loss = criterion_fine_tune(out, target)

                #########################we implement FedProx Here###########################
                fed_prox_reg = 0.0
                for param_index, param in enumerate(matched_cnn.parameters()):
                    fed_prox_reg += ((mu / 2) * torch.norm((param - global_weight_collector[param_index])) ** 2)
                loss += fed_prox_reg
                ##############################################################################

                epoch_loss_collector.append(loss.item())

                loss.backward()
                optimizer_fine_tune.step()

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

    train_acc = compute_accuracy(matched_cnn, train_dl_local, device=device)
    test_acc, conf_matrix = compute_accuracy(matched_cnn, test_dl_local, get_confusion_matrix=True, device=device)

    logger.info('>> Training accuracy after local retrain: %f' % train_acc)
    logger.info('>> Test accuracy after local retrain: %f' % test_acc)
    # logger.info('>> Test accuracy after local retrain: %f' % test_acc)
    return matched_cnn

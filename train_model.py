import torch
import torch.distributed as dist
from utils import *
import torch.nn as nn
import numpy as np
import cupy as cp
import tensorly

CUDA = torch.cuda.is_available()
###dist.init_process_group(backend='nccl', init_method='env://')

def compute_tucker_decomposition(X, core_shape):
    """
    Computes Tucker decomposition of `X` with `core_shape`, and caches result.

    Args:
        X: Arbitrary tensor of type `np.ndarray`.
        core_shape: List of integers that is the same dimension as `X`.
        output_path: Path where Tucker decompositions for `X` are cached.
    """

    ###if os.path.exists(filename):
    ###    solve_result = read_tucker_decomposition_solve_result_from_file(filename)
    ###    return solve_result

    input_shape = X.shape
    core_shape = tuple(core_shape)
    '''
    num_tucker_params = get_num_tucker_params(X, core_shape)
    compression_ratio = num_tucker_params / X.size'''

    init = 'random'
    n_iter_max = 20
    tol = 1e-20
    core, factors = tensorly.decomposition.tucker(X, rank=core_shape,
                                                  init=init,
                                                  n_iter_max=n_iter_max,
                                                  tol=tol
                                                  )
    X_hat = tensorly.tucker_to_tensor((core, factors))
    loss = np.linalg.norm(X - X_hat) ** 2
    rre = loss / np.linalg.norm(X) ** 2
    return core, factors


def centering_matrix(m):
    J_m = torch.eye(m) - (torch.ones([m, 1]) @ torch.ones([1, m])) * (1.0 / m)
    return J_m


# Taylor expansion
def matrix_log(Q, order=4):
    n = Q.shape[0]
    Q = Q - torch.eye(n).detach().to(Q.device)
    cur = Q
    res = torch.zeros_like(Q).detach().to(Q.device)
    for k in range(1, order + 1):
        if k % 2 == 1:
            res = res + cur * (1. / float(k))
        else:
            res = res - cur * (1. / float(k))
        cur = cur @ Q

    return res


class GatherLayer(torch.autograd.Function):
    """
    Gather tensors from all workers with support for backward propagation:
    This implementation does not cut the gradients as torch.distributed.all_gather does.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


def gather_from_all(tensor: torch.Tensor) -> torch.Tensor:
    """
    Similar to classy_vision.generic.distributed_util.gather_from_all
    except that it does not cut the gradients
    """
    if tensor.ndim == 0:
        # 0 dim tensors cannot be gathered. so unsqueeze
        tensor = tensor.unsqueeze(0)

    gathered_tensors = GatherLayer.apply(tensor)

    gathered_tensor = torch.cat(gathered_tensors, 0)

    return gathered_tensor


def mce_loss_func(p, z, lamda=1., mu=1., order=4, align_gamma=0.003, correlation=True, logE=False):
    ###p = gather_from_all(p)
    ###z = gather_from_all(z)
    '''p = p.T###
    z = z.T###'''
    p = F.normalize(p)
    z = F.normalize(z)

    m = z.shape[0]
    n = z.shape[1]
    # print(m, n)
    J_m = centering_matrix(m).detach().to(z.device)

    if correlation:
        P = lamda * torch.eye(n).to(z.device)
        Q = (1. / m) * (p.T @ J_m @ z) + mu * torch.eye(n).to(z.device)
    else:
        P = (1. / m) * (p.T @ J_m @ p) + mu * torch.eye(n).to(z.device)
        Q = (1. / m) * (z.T @ J_m @ z) + mu * torch.eye(n).to(z.device)

    return torch.trace(- P @ matrix_log(Q, order))


def train_one_epoch(data_loader, net, loss_fn, optimizer):
    net.train()
    tl = Averager()
    pred_train = []
    act_train = []
    for i, (x_batch, y_batch) in enumerate(data_loader):
        s = [64, 1, 10, 512]
        ### = np.asarray(x_batch)###
        ###x_batch1, factors = compute_tucker_decomposition(x_batch, s)
        if CUDA:
            x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
        out = net(x_batch)
        loss = loss_fn(out, y_batch)
        _, pred = torch.max(out, 1)
        pred_train.extend(pred.data.tolist())
        act_train.extend(y_batch.data.tolist())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tl.add(loss.item())
    return tl.item(), pred_train, act_train


def train_one_epoch_with_DA(data_loader, target_dataloader, net, epoch, loss_fn, optimizer, lambda1=1e-5):
    net.train()
    tl = Averager()
    pred_train = []
    act_train = []
    for i, (x_batch, y_batch) in enumerate(data_loader):
        p = float(i + epoch * len(data_loader)) / 12 / len(data_loader)
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        if CUDA:
            x_batch, y_batch = x_batch.cuda(), y_batch.cuda()

        if x_batch.size(0) == 1:
            # To avoid single sample issue
            x_batch = torch.cat((x_batch, x_batch), dim=0)
            y_batch = torch.cat((y_batch, y_batch), dim=0)

        out, domain_output = net(x_batch, alpha)###LGG
        loss = loss_fn(out, y_batch)###LGG
        y, y_rec, outm, out, domain_output, sdta = net(x_batch, alpha)
        loss_ce = loss_fn(out, y_batch)
        mse_f = nn.MSELoss()
        loss_rec = mse_f(y, y_rec)
        cos_f = nn.CosineSimilarity()
        loss_ort = 0
        ll = len(outm)
        for k in range(ll):
            for j in range(ll):
                if k == j:
                    continue
                else:
                    loss_ort += cos_f(outm[k], outm[j])
        loss_ort /= (ll * (ll - 1))
        loss_ort = torch.mean(loss_ort, dim=0)

        domain_label = torch.zeros(domain_output.shape[0]).long().cuda()
        err_s_domain = F.cross_entropy(domain_output, domain_label)
        # Training model using target data
        target_dataloader_len = len(target_dataloader)
        if i % target_dataloader_len == 0:
            target_dataloader_iter = iter(target_dataloader)
        data_target = target_dataloader_iter.__next__()
        data_target = data_target[0].cuda()
        ###target_output, domain_output = net(data_target, alpha)
        y, y_rec, outm, target_output, domain_output, tdta = net(data_target, alpha)  ###target_output, domain_output
        domain_label = torch.ones(domain_output.shape[0]).long().cuda()
        err_t_domain = F.cross_entropy(domain_output, domain_label)
        loss_d = lambda1 * (err_t_domain + err_s_domain)  ###lambda   mce_lambd

        mce_lambd = 0.5
        mce_mu = 0.5
        gamma = 1.0
        mce_order = 4
        align_gamma = 0.003
        if tdta.size()[0] == sdta.size()[0]:
            loss_ta = (mce_loss_func(tdta, sdta, correlation=True, logE=False, lamda=mce_lambd, mu=mce_mu,
                                     order=mce_order, align_gamma=align_gamma)
                       + gamma * mce_loss_func(tdta, sdta, correlation=False, logE=False, lamda=mce_lambd,
                                               mu=mce_mu, order=mce_order, align_gamma=align_gamma))
        else:
            loss_ta = 0
        gd = 0.2
        grec = 0.2
        gort = 0.2
        gta = 0.2
        gce = 1 - gd - grec - gort - gta
        loss = gce * loss_ce + gd * loss_d + grec * loss_rec + gort * loss_ort + gta * loss_ta / 100
        ###loss = loss + loss_d###LGG

        _, pred = torch.max(out, 1)
        pred_train.extend(pred.data.tolist())
        act_train.extend(y_batch.data.tolist())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tl.add(loss.item())
    return tl.item(), pred_train, act_train


def predict(data_loader, net, loss_fn):
    net.eval()
    pred_val = []
    act_val = []
    vl = Averager()
    with torch.no_grad():
        for i, (x_batch, y_batch) in enumerate(data_loader):
            if CUDA:
                x_batch, y_batch = x_batch.cuda(), y_batch.cuda()

            ###_, _, _, out, _, _ = net(x_batch)  ###out, _
            out = net(x_batch)
            loss = loss_fn(out, y_batch)
            _, pred = torch.max(out, 1)
            vl.add(loss.item())
            pred_val.extend(pred.data.tolist())
            act_val.extend(y_batch.data.tolist())
    return vl.item(), pred_val, act_val


def set_up(args):
    set_gpu(args.gpu)
    ensure_path(args.save_path)
    torch.manual_seed(args.random_seed)
    torch.backends.cudnn.deterministic = True


def train(args, data_train, label_train, data_val, label_val, data_test, label_test, subject, fold):  ###train with test
    seed_all(args.random_seed)
    save_name = '_sub' + str(subject) + '_fold' + str(fold)
    set_up(args)

    train_loader = get_dataloader(data_train, label_train, args.batch_size)

    val_loader = get_dataloader(data_val, label_val, args.batch_size)

    test_loader = get_dataloader(data_test, label_test, args.batch_size)

    model = get_model(args)
    if CUDA:
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    if args.LS:
        loss_fn = LabelSmoothing(args.LS_rate)
    else:
        loss_fn = nn.CrossEntropyLoss()

    def save_model(name):
        previous_model = osp.join(args.save_path, '{}.pth'.format(name))
        if os.path.exists(previous_model):
            os.remove(previous_model)
        torch.save(model.state_dict(), osp.join(args.save_path, '{}.pth'.format(name)))

    trlog = {}
    trlog['args'] = vars(args)
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['train_acc'] = []
    trlog['val_acc'] = []
    trlog['max_acc'] = 0.0
    trlog['F1'] = 0.0

    timer = Timer()
    patient = args.patient
    counter = 0

    for epoch in range(1, args.max_epoch + 1):

        loss_train, pred_train, act_train = train_one_epoch(
            data_loader=train_loader, net=model, loss_fn=loss_fn, optimizer=optimizer)
        acc_train, f1_train, _ = get_metrics(y_pred=pred_train, y_true=act_train)
        print('epoch {}, loss={:.4f} acc={:.4f} f1={:.4f}'
              .format(epoch, loss_train, acc_train, f1_train))

        loss_val, pred_val, act_val = predict(
            data_loader=val_loader, net=model, loss_fn=loss_fn)
        acc_val, f1_val, _ = get_metrics(y_pred=pred_val, y_true=act_val)
        print('epoch {}, val, loss={:.4f} acc={:.4f} f1={:.4f}'.
              format(epoch, loss_val, acc_val, f1_val))

        loss, pred, act = predict(
            data_loader=test_loader, net=model, loss_fn=loss_fn)
        acc, f1, cm = get_metrics(y_pred=pred, y_true=act)
        print('>>> Test:  loss={:.4f} acc={:.4f} f1={:.4f}'.format(loss, acc, f1))

        if acc >= trlog['max_acc']:  ###acc_val>=
            trlog['max_acc'] = acc  ###acc_val
            trlog['F1'] = f1  ###f1_val
            save_model('max-acc')  ###candidate
            counter = 0
        else:
            counter += 1
            if counter >= patient:
                print('early stopping')
                break

        trlog['train_loss'].append(loss_train)
        trlog['train_acc'].append(acc_train)
        trlog['val_loss'].append(loss_val)
        trlog['val_acc'].append(acc_val)

        print('ETA:{}/{} SUB:{} FOLD:{}'.format(timer.measure(), timer.measure(epoch / args.max_epoch),
                                                subject, fold))
    # save the training log file
    save_name = 'trlog' + save_name
    experiment_setting = 'T_{}_pool_{}'.format(args.T, args.pool)
    save_path = osp.join(args.save_path, experiment_setting, 'log_train')
    ensure_path(save_path)
    torch.save(trlog, osp.join(save_path, save_name))

    return trlog['max_acc'], trlog['F1']


def train_with_DA(args, data_train, label_train, data_val, label_val, data_test, label_test, subject,
                  fold):  ###train with test
    seed_all(args.random_seed)
    save_name = '_sub' + str(subject) + '_fold' + str(fold)
    set_up(args)

    train_loader = get_dataloader(data_train, label_train, args.batch_size)

    val_loader = get_dataloader(data_val, label_val, 1)

    test_loader = get_dataloader(data_test, label_test, args.batch_size)

    model = get_model(args)
    if CUDA:
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    if args.LS:
        loss_fn = LabelSmoothing(args.LS_rate)
    else:
        loss_fn = nn.CrossEntropyLoss()

    def save_model(name):
        previous_model = osp.join(args.save_path, '{}.pth'.format(name))
        if os.path.exists(previous_model):
            os.remove(previous_model)
        torch.save(model.state_dict(), osp.join(args.save_path, '{}.pth'.format(name)))

    trlog = {}
    trlog['args'] = vars(args)
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['train_acc'] = []
    trlog['val_acc'] = []
    trlog['max_acc'] = 0.0
    trlog['F1'] = 0.0

    timer = Timer()
    patient = args.patient
    counter = 0

    for epoch in range(1, args.max_epoch + 1):

        loss_train, pred_train, act_train = train_one_epoch_with_DA(  ###train_one_epoch
            data_loader=train_loader,
            target_dataloader=test_loader,
            epoch=epoch, net=model,
            loss_fn=loss_fn, optimizer=optimizer)
        acc_train, f1_train, _ = get_metrics(y_pred=pred_train, y_true=act_train)
        print('epoch {}, loss={:.4f} acc={:.4f} f1={:.4f}'
              .format(epoch, loss_train, acc_train, f1_train))

        loss_val, pred_val, act_val = predict(
            data_loader=val_loader, net=model, loss_fn=loss_fn)
        acc_val, f1_val, _ = get_metrics(y_pred=pred_val, y_true=act_val)
        print('epoch {}, val, loss={:.4f} acc={:.4f} f1={:.4f}'.
              format(epoch, loss_val, acc_val, f1_val))

        loss, pred, act = predict(
            data_loader=test_loader, net=model, loss_fn=loss_fn)
        acc, f1, cm = get_metrics(y_pred=pred, y_true=act)
        print('>>> Test:  loss={:.4f} acc={:.4f} f1={:.4f}'.format(loss, acc, f1))

        if acc >= trlog['max_acc']:  ###acc_val>=
            trlog['max_acc'] = acc  ###acc_val
            trlog['F1'] = f1  ###f1_val
            save_model('max-acc')  ###candidate
            counter = 0
        else:
            counter += 1
            if counter >= patient:
                print('early stopping')
                break

        trlog['train_loss'].append(loss_train)
        trlog['train_acc'].append(acc_train)
        trlog['val_loss'].append(loss_val)
        trlog['val_acc'].append(acc_val)

        print('ETA:{}/{} SUB:{} FOLD:{}'.format(timer.measure(), timer.measure(epoch / args.max_epoch),
                                                subject, fold))
    # save the training log file
    save_name = 'trlog' + save_name
    experiment_setting = 'T_{}_pool_{}'.format(args.T, args.pool)
    save_path = osp.join(args.save_path, experiment_setting, 'log_train')
    ensure_path(save_path)
    torch.save(trlog, osp.join(save_path, save_name))

    return trlog['max_acc'], trlog['F1']


def test(args, data, label, reproduce, subject, fold):
    set_up(args)
    seed_all(args.random_seed)
    test_loader = get_dataloader(data, label, args.batch_size)

    model = get_model(args)
    if CUDA:
        model = model.cuda()
    loss_fn = nn.CrossEntropyLoss()

    if reproduce:
        model_name_reproduce = 'sub' + str(subject) + '_fold' + str(fold) + '.pth'
        data_type = 'model_{}_{}'.format(args.data_format, args.label_type)
        experiment_setting = 'T_{}_pool_{}'.format(args.T, args.pool)
        load_path_final = osp.join(args.save_path, experiment_setting, data_type, model_name_reproduce)
        model.load_state_dict(torch.load(load_path_final))
    else:
        model.load_state_dict(torch.load(args.load_path))  ###load_path_final
    loss, pred, act = predict(
        data_loader=test_loader, net=model, loss_fn=loss_fn
    )
    acc, f1, cm = get_metrics(y_pred=pred, y_true=act)
    print('>>> Test:  loss={:.4f} acc={:.4f} f1={:.4f}'.format(loss, acc, f1))
    return acc, pred, act


def combine_train(args, data, label, subject, fold, target_acc):
    save_name = '_sub' + str(subject) + '_fold' + str(fold)
    set_up(args)
    seed_all(args.random_seed)
    train_loader = get_dataloader(data, label, args.batch_size)
    model = get_model(args)
    if CUDA:
        model = model.cuda()
    model.load_state_dict(torch.load(args.load_path))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate * 1e-1)

    if args.LS:
        loss_fn = LabelSmoothing(args.LS_rate)
    else:
        loss_fn = nn.CrossEntropyLoss()

    def save_model(name):
        previous_model = osp.join(args.save_path, '{}.pth'.format(name))
        if os.path.exists(previous_model):
            os.remove(previous_model)
        torch.save(model.state_dict(), osp.join(args.save_path, '{}.pth'.format(name)))

    trlog = {}
    trlog['args'] = vars(args)
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['train_acc'] = []
    trlog['val_acc'] = []
    trlog['max_acc'] = 0.0

    timer = Timer()

    for epoch in range(1, args.max_epoch_cmb + 1):
        loss, pred, act = train_one_epoch(
            data_loader=train_loader, net=model, loss_fn=loss_fn, optimizer=optimizer
        )
        acc, f1, _ = get_metrics(y_pred=pred, y_true=act)
        print('Stage 2 : epoch {}, loss={:.4f} acc={:.4f} f1={:.4f}'
              .format(epoch, loss, acc, f1))

        if acc >= target_acc or epoch == args.max_epoch_cmb:
            print('early stopping!')
            save_model('final_model')
            # save model here for reproduce
            model_name_reproduce = 'sub' + str(subject) + '_fold' + str(fold) + '.pth'
            data_type = 'model_{}_{}'.format(args.data_format, args.label_type)
            experiment_setting = 'T_{}_pool_{}'.format(args.T, args.pool)
            save_path = osp.join(args.save_path, experiment_setting, data_type)
            ensure_path(save_path)
            model_name_reproduce = osp.join(save_path, model_name_reproduce)
            torch.save(model.state_dict(), model_name_reproduce)
            break

        trlog['train_loss'].append(loss)
        trlog['train_acc'].append(acc)

        print('ETA:{}/{} SUB:{} TRIAL:{}'.format(timer.measure(), timer.measure(epoch / args.max_epoch),
                                                 subject, fold))

    save_name = 'trlog_comb' + save_name
    experiment_setting = 'T_{}_pool_{}'.format(args.T, args.pool)
    save_path = osp.join(args.save_path, experiment_setting, 'log_train_cmb')
    ensure_path(save_path)
    torch.save(trlog, osp.join(save_path, save_name))

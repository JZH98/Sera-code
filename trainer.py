from utils import *
import copy
import torch.nn.functional as F

CUDA = torch.cuda.is_available()

class Trainer:
    def __init__(self, args, subject, fold_ID):
        self.args = args
        self.set_up()
        self.best_epoch_info = None
        self.model = None
        self.current_subject = subject
        self.current_fold_ID = fold_ID
        self.model_name_reproduce = 'sub_{}_fold_{}'.format(subject, fold_ID)
        self.current_data_info = 'model_{}_{}'.format(args.data_format, args.label_type)
        self.best_model_saved = False

    def save_model(self):
        ensure_path(osp.join(self.args.save_path, self.current_data_info))
        model_name_reproduce = osp.join(self.current_data_info, self.model_name_reproduce)
        torch.save(self.best_epoch_info['model_weights'], osp.join(self.args.save_path, model_name_reproduce + '.pth'))


    def set_up(self):
        if CUDA:
            set_gpu(self.args.gpu)
        ensure_path(self.args.save_path)
        torch.manual_seed(self.args.random_seed)
        torch.backends.cudnn.deterministic = True

    def centering_matrix(self, m):
        J_m = torch.eye(m) - (torch.ones([m, 1]) @ torch.ones([1, m])) * (1.0 / m)
        return J_m

    # Taylor expansion
    def matrix_log(self, Q, order=4):
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

    def mce_loss_func(self, p, z, lamda=1., mu=1., order=4, align_gamma=0.003, correlation=True, logE=False):
        ###p = gather_from_all(p)
        ###z = gather_from_all(z)
        '''p = p.T###
        z = z.T###'''
        p = F.normalize(p)
        z = F.normalize(z)

        m = z.shape[0]
        n = z.shape[1]
        # print(m, n)
        J_m = self.centering_matrix(m).detach().to(z.device)

        if correlation:
            P = lamda * torch.eye(n).to(z.device)
            Q = (1. / m) * (p.T @ J_m @ z) + mu * torch.eye(n).to(z.device)
        else:
            P = (1. / m) * (p.T @ J_m @ p) + mu * torch.eye(n).to(z.device)
            Q = (1. / m) * (z.T @ J_m @ z) + mu * torch.eye(n).to(z.device)

        return torch.trace(- P @ self.matrix_log(Q, order))

    def covariance_matrix(self, X):
        # X is of shape (batch_size, n, m)
        # Center the data
        X_centered = X - X.mean(dim=1, keepdim=True)
        # Compute covariance matrix for each sample in the batch
        cov_matrix = torch.matmul(X_centered.transpose(1, 2), X_centered) / (X_centered.size(1) - 1)
        return cov_matrix

    def temporal_alignment_loss(self, P, Q):
        # P and Q: b, seq, h
        # Compute batch covariance matrices
        cov_P = self.covariance_matrix(P)#where label == 1
        cov_Q = self.covariance_matrix(Q)

        diff = cov_P - cov_Q
        # Compute the Frobenius norm of the differences for each pair
        loss = torch.norm(diff, p='fro', dim=(1, 2))
        # Average loss over the batch
        loss = torch.mean(loss)
        return loss

    def train_one_epoch(self, data_loader, net, loss_fn, optimizer):
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

    def train_one_epoch_with_DA(self, data_loader, target_dataloader, net, epoch, loss_fn, optimizer, lambda1=1e-5):
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
            if self.args.model ==  'TransDANN':
                out, domain_output = net(x_batch, alpha)
                loss = loss_fn(out, y_batch)

                domain_label = torch.zeros(domain_output.shape[0]).long().cuda()
                err_s_domain = F.cross_entropy(domain_output, domain_label)
                # Training model using target data
                target_dataloader_len = len(target_dataloader)
                if i % target_dataloader_len == 0:
                    target_dataloader_iter = iter(target_dataloader)
                data_target = target_dataloader_iter.__next__()
                data_target = data_target[0].cuda()
                target_output, domain_output = net(data_target, alpha)
                domain_label = torch.ones(domain_output.shape[0]).long().cuda()
                err_t_domain = F.cross_entropy(domain_output, domain_label)
                loss = loss + lambda1 * (err_t_domain + err_s_domain)  ###lambda

                _, pred = torch.max(out, 1)
                pred_train.extend(pred.data.tolist())
                act_train.extend(y_batch.data.tolist())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                tl.add(loss.item())
                continue

            y, y_rec, outm, out, domain_output, sdta, clsf = net(x_batch, alpha)

            loss_ce = loss_fn(out, y_batch)

            mse_f = nn.MSELoss()
            loss_rec = mse_f(y, y_rec)
            cos_f = nn.CosineSimilarity()
            loss_ort = 0
            ll = outm.shape[1]

            domain_label = torch.zeros(domain_output.shape[0]).long().cuda()
            err_s_domain = F.cross_entropy(domain_output, domain_label)
            # Training model using target data
            target_dataloader_len = len(target_dataloader)
            if i % target_dataloader_len == 0:
                target_dataloader_iter = iter(target_dataloader)
            data_target = target_dataloader_iter.__next__()
            data_target = data_target[0].cuda()
            y_t, y_rec_t, outm_t, target_output, domain_output, tdta, clsf = net(data_target,
                                                                            alpha)  ###target_output, domain_output

            for k in range(ll):
                for j in range(ll):
                    if k == j:
                        continue
                    else:
                        loss_ort += cos_f(outm[k], outm[j])
                        loss_ort += cos_f(outm_t[k], outm_t[j])
            loss_ort /= (ll * (ll - 1))
            loss_ort = torch.mean(loss_ort, dim=0)

            domain_label = torch.ones(domain_output.shape[0]).long().cuda()
            err_t_domain = F.cross_entropy(domain_output, domain_label)
            loss_d = err_t_domain + err_s_domain  ###lambda1   mce_lambd

            if tdta.size()[0] == sdta.size()[0]:
                loss_ta = self.temporal_alignment_loss(tdta, sdta)
            else:
                loss_ta = 0
            gd = 0.01
            grec = 0.001
            gort = 0.1
            gta = 0.01
            gce = 1 - gd - grec - gort - gta
            loss = gce * loss_ce + gd * loss_d + grec * loss_rec + gort * loss_ort + gta * loss_ta

            _, pred = torch.max(out, 1)
            pred_train.extend(pred.data.tolist())
            act_train.extend(y_batch.data.tolist())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tl.add(loss.item())
        return tl.item(), pred_train, act_train

    def predict(self, data_loader, net, loss_fn, subject=0):
        net.eval()
        pred_val = []
        act_val = []
        vl = Averager()
        with torch.no_grad():
            for i, (x_batch, y_batch) in enumerate(data_loader):
                if CUDA:
                    x_batch, y_batch = x_batch.cuda(), y_batch.cuda()

                if self.args.model == 'SERA':
                    y, y_rec, outm, out, domain_output, sdta, clsf = net(x_batch)  ###out, _
                    '''if y_batch.size(0) == self.args.batch_size:
                        outm_cpu = outm.detach().cpu().numpy()
                        np.save('./featuresForTSNE/outm'+ str(subject)+ '-' + str(i) +'.npy', arr=outm_cpu)
                        y_batch_cpu = y_batch.detach().cpu().numpy()
                        np.save('./featuresForTSNE/y_batch' + str(subject)+ '-' + str(i) + '.npy', arr=y_batch_cpu)
                        x_batch_cpu = x_batch.detach().cpu().numpy()
                        np.save('./featuresForTSNE/x_batch' + str(subject) + '-' + str(i) + '.npy', arr=x_batch_cpu)
                        y_cpu = y.detach().cpu().numpy()
                        np.save('./featuresForTSNE/y' + str(subject) + '-' + str(i) + '.npy', arr=y_cpu)
                        sdta_cpu = sdta.detach().cpu().numpy()
                        np.save('./featuresForTSNE/sdta' + str(subject) + '-' + str(i) + '.npy', arr=sdta_cpu)
                        clsf_cpu = clsf.detach().cpu().numpy()
                        np.save('./featuresForTSNE/clsf' + str(subject) + '-' + str(i) + '.npy', arr=clsf_cpu)'''####

                elif self.args.model == 'TransDANN':
                    out, _ = net(x_batch)
                else:
                    out = net(x_batch)

                loss = loss_fn(out, y_batch)
                _, pred = torch.max(out, 1)
                vl.add(loss.item())
                pred_val.extend(pred.data.tolist())
                act_val.extend(y_batch.data.tolist())
        return vl.item(), pred_val, act_val

    def train(self, args, data_train, label_train, data_val, label_val, data_test, label_test, subject,
              fold):  ###train with test
        seed_all(args.random_seed)
        save_name = '_sub' + str(subject) + '_fold' + str(fold)
        self.set_up()

        train_loader = get_dataloader(data_train, label_train, args.batch_size)

        val_loader = get_dataloader(data_val, label_val, args.batch_size)

        test_loader = get_dataloader(data_test, label_test, args.batch_size)

        self.model = get_model(self.args)
        if CUDA:
            self.model = self.model.cuda()

        if self.best_epoch_info is None:
            self.best_epoch_info = {
                'model_weights': copy.deepcopy(self.model.state_dict()),
                'loss': 1e10,
                'acc': -1e10,
                'f1': -1e10
            }

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.learning_rate)

        if args.LS:
            loss_fn = LabelSmoothing(args.LS_rate)
        else:
            loss_fn = nn.CrossEntropyLoss()


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

            loss_train, pred_train, act_train = self.train_one_epoch(
                data_loader=train_loader, net=self.model, loss_fn=loss_fn, optimizer=optimizer)
            acc_train, f1_train, _ = get_metrics(y_pred=pred_train, y_true=act_train)
            print('epoch {}, loss={:.4f} acc={:.4f} f1={:.4f}'
                  .format(epoch, loss_train, acc_train, f1_train))

            loss_val, pred_val, act_val = self.predict(
                data_loader=val_loader, net=self.model, loss_fn=loss_fn)
            acc_val, f1_val, _ = get_metrics(y_pred=pred_val, y_true=act_val)
            print('epoch {}, val, loss={:.4f} acc={:.4f} f1={:.4f}'.
                  format(epoch, loss_val, acc_val, f1_val))

            loss, pred, act = self.predict(
                data_loader=test_loader, net=self.model, loss_fn=loss_fn)
            acc, f1, cm = get_metrics(y_pred=pred, y_true=act)

            if acc >= trlog['max_acc']:
                trlog['max_acc'] = acc
                trlog['F1'] = f1
                self.best_epoch_info = {
                    'model_weights': copy.deepcopy(self.model.state_dict()),
                    'loss': loss,
                    'acc': acc,
                    'f1': f1
                }
                self.best_model_saved = True
                print("Model saved!")
                counter = 0
            else:
                counter += 1
                if epoch == self.args.max_epoch and not self.best_model_saved:
                    self.best_epoch_info = {
                        'model_weights': copy.deepcopy(self.model.state_dict()),
                        'loss': loss,
                        'acc': acc,
                        'f1': f1
                    }
                    self.best_model_saved = True
                    trlog['max_acc'] = acc

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

    def train_with_DA(self, args, data_train, label_train, data_val, label_val, data_test, label_test, subject,
                      fold):
        seed_all(args.random_seed)
        save_name = '_sub' + str(subject) + '_fold' + str(fold)
        self.set_up()

        train_loader = get_dataloader(data_train, label_train, args.batch_size)

        val_loader = get_dataloader(data_val, label_val, 1)

        test_loader = get_dataloader(data_test, label_test, args.batch_size)

        self.model = get_model(self.args)
        if CUDA:
            self.model = self.model.cuda()

        if self.best_epoch_info is None:
            self.best_epoch_info = {
                'model_weights': copy.deepcopy(self.model.state_dict()),
                'loss': 1e10,
                'acc': -1e10,
                'f1': -1e10
            }

        optimizer = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate)

        if args.LS:
            loss_fn = LabelSmoothing(args.LS_rate)
        else:
            loss_fn = nn.CrossEntropyLoss()



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

            loss_train, pred_train, act_train = self.train_one_epoch_with_DA(  ###train_one_epoch
                data_loader=train_loader,
                target_dataloader=test_loader,
                epoch=epoch, net=self.model,
                loss_fn=loss_fn, optimizer=optimizer)
            acc_train, f1_train, _ = get_metrics(y_pred=pred_train, y_true=act_train)
            print('epoch {}, loss={:.4f} acc={:.4f} f1={:.4f}'
                  .format(epoch, loss_train, acc_train, f1_train))

            loss_val, pred_val, act_val = self.predict(
                data_loader=val_loader, net=self.model, loss_fn=loss_fn)
            acc_val, f1_val, _ = get_metrics(y_pred=pred_val, y_true=act_val)
            print('epoch {}, val, loss={:.4f} acc={:.4f} f1={:.4f}'.
                  format(epoch, loss_val, acc_val, f1_val))

            loss, pred, act = self.predict(
                data_loader=test_loader, net=self.model, loss_fn=loss_fn, subject=subject)
            acc, f1, cm = get_metrics(y_pred=pred, y_true=act)
            print('>>> Test:  loss={:.4f} acc={:.4f} f1={:.4f}'.format(loss, acc, f1))

            if acc >= trlog['max_acc']:
                trlog['max_acc'] = acc
                trlog['F1'] = f1
                self.best_epoch_info = {
                    'model_weights': copy.deepcopy(self.model.state_dict()),
                    'loss': loss,
                    'acc': acc,
                    'f1': f1
                }
                self.best_model_saved = True
                self.save_model()
                print("Model saved!")
                counter = 0
            else:
                counter += 1
                if epoch == self.args.max_epoch and not self.best_model_saved:
                    self.best_epoch_info = {
                        'model_weights': copy.deepcopy(self.model.state_dict()),
                        'loss': loss,
                        'acc': acc,
                        'f1': f1
                    }
                    self.best_model_saved = True
                    self.save_model()
                    trlog['max_acc'] = acc

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

    def test(self, args, data, label, reproduce, subject, fold):
        self.set_up()
        seed_all(args.random_seed)
        test_loader = get_dataloader(data, label, args.batch_size)

        self.model = get_model(self.args)
        if CUDA:
            self.model = self.model.cuda()
        loss_fn = nn.CrossEntropyLoss()

        if reproduce:
            load_path_final = osp.join(self.args.save_path, self.current_data_info, self.model_name_reproduce)
            self.model.load_state_dict(torch.load(load_path_final))
        else:
            self.model.load_state_dict(self.best_epoch_info['model_weights'])

        loss, pred, act = self.predict(
            data_loader=test_loader, net=self.model,
            loss_fn=loss_fn
        )
        acc, f1, cm = get_metrics(y_pred=pred, y_true=act)
        print('>>> Test:  loss={:.4f} acc={:.4f} f1={:.4f}'.format(loss, acc, f1))
        return acc, pred, act




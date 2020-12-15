import torch
import torch.optim as optim
import numpy as np
import Validation
from sklearn.neighbors import BallTree
import tqdm


class DeepKnn():
    def __init__(self, K=5, M=5, alpha=0.1):
        self.embeddings = [None, None]
        self.nn = [None, None]
        self.scores = []
        self.K = K
        self.M = M
        self.alpha = alpha

    def test(self, test_loader, net):

        idx_label_targets = []
        net.eval()
        tbar = tqdm.tqdm(iter(test_loader), total=len(test_loader), position=0, leave=True)
        with torch.no_grad():
            for data in tbar:
                inputs, targets, idx = data
                inputs = inputs.cuda()

                outputs = net(inputs)

                norm_dist, _ = self.nn[0].query(outputs.detach().cpu().numpy(), k=self.K)
                anom_dist, _ = self.nn[1].query(outputs.detach().cpu().numpy(), k=self.K)

                all_n = np.concatenate((norm_dist, anom_dist), axis=1)
                all_k = all_n.argsort()[:, :self.K] > self.K
                labels = np.sum(all_k, 1) > int(self.K / 2)

                idx_label_targets += list(zip(targets.numpy().tolist(),
                                              labels.tolist()))

        targets, labels = zip(*idx_label_targets)
        targets = np.array(targets)
        labels = np.array(labels)

        precision, recall, fscore, mcc, val_acc = Validation.evaluate(targets, labels)
        self.scores.append((precision, recall, fscore, mcc, val_acc))
        print('\tV_acc  %.3f\tPrecis %.3f\tRecall %.3f\tFscore %.3f\tMCC %.3f' % (
        val_acc, precision, recall, fscore, mcc))

    def train(self, train_loader, net, lr, epochs, test_loader, weight_decay=1e-6):

        # Set optimizer (Adam optimizer for now)
        optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

        # Training
        for epoch in range(epochs):

            net.eval()
            with torch.no_grad():
                self.embeddings[0] = net(train_loader.dataset.data_0.cuda()).cpu().numpy()
                self.embeddings[1] = net(train_loader.dataset.data_1.cuda()).cpu().numpy()
                self.nn[0] = BallTree(self.embeddings[0], leaf_size=self.K + self.M + 2)
                self.nn[1] = BallTree(self.embeddings[1], leaf_size=self.K + self.M + 2)

            net.train()
            epoch_loss = 0.0
            n_batches = 0

            optimizer.zero_grad()

            outputs = net(train_loader.dataset.data.cuda())
            outputs_sep = [outputs[train_loader.dataset.labels == 0],
                           outputs[train_loader.dataset.labels == 1]]
            all_emb = outputs.detach().cpu().numpy()
            losses = []

            tbar = tqdm.tqdm(iter(train_loader), total=len(train_loader), position=0, leave=True)
            for _, targets, idx in tbar:

                same_vecs = []
                diff_vecs = []
                for i, t in zip(idx, targets):
                    _, same_idx = self.nn[t.item()].query(np.expand_dims(all_emb[i], 0), k=self.K + 1)
                    _, diff_idx = self.nn[1 - t.item()].query(np.expand_dims(all_emb[i], 0), k=self.M)

                    same_vecs.append(torch.unsqueeze(outputs_sep[t.item()][same_idx[:, 1:]], dim=0))
                    diff_vecs.append(torch.unsqueeze(outputs_sep[1 - t.item()][diff_idx], dim=0))

                same_vecs = torch.cat(same_vecs)
                diff_vecs = torch.cat(diff_vecs)

                out_unsqz = torch.unsqueeze(outputs[idx], dim=1)
                same_dist = torch.norm(out_unsqz - same_vecs, dim=2, p=None)
                diff_dist = torch.norm(out_unsqz - diff_vecs, dim=2, p=None)

                loss = torch.zeros(idx.shape[0]).cuda()
                constant = torch.Tensor([0]).cuda()
                for i in range(self.K):
                    for j in range(self.M):
                        val = same_dist[:, i] + self.alpha - diff_dist[:, j]
                        loss += torch.where(val > 0, val, constant)

                loss = loss / self.K / self.M
                loss = torch.mean(loss)
                losses.append(torch.unsqueeze(loss, dim=0))

                epoch_loss += loss.item()
                n_batches += 1

            losses = torch.mean(torch.cat(losses))
            losses.backward()
            optimizer.step()

            self.test(test_loader, net)
            # log epoch statistics
            print(f'Train Loss: {epoch_loss / n_batches:.6f}')
            # thresh = eval(test_loader, net, 0, bval)

        print('Finished training.')

# # built-in modules
import time
from logging import Logger
import os
# # Torch modules
import torch
from torch.nn.functional import cross_entropy, mse_loss
from torch.nn.utils import clip_grad_norm_
# # internal imports
from .model import AttentionModel
from .utils import get_grad_norms, get_ior_match, plot_all


def pixel_error(predictions: torch.Tensor, targets: torch.Tensor, reduction: str = 'mean', donorm: bool = True) -> float:
    if donorm:
        predictions = (predictions + 1.0) / 2.0
        targets = (targets + 1.0) / 2.0
    corrects = (predictions - targets).square().sum()
    return corrects / targets.numel() if reduction == 'mean' else corrects * targets.size(0) / targets.numel()


def normed_acc(predictions: torch.Tensor, targets: torch.Tensor, reduction: str = 'mean', donorm: bool = True) -> float:
    n = targets.size(0)
    if donorm:
        predictions = (predictions + 1.0) / 2.0
        targets = (targets + 1.0) / 2.0
    corrects = 0.0
    corrects = (predictions * targets > 0.5).sum() / targets.sum()
    corrects += ((1.0 - predictions) * (1.0 - targets) > 0.5).sum() / (1.0 - targets).sum()
    return corrects / 2.0 if reduction == 'mean' else corrects * n / 2.0

class AttentionTrain:
    def __init__(self, 
                 model: AttentionModel, 
                 optimizer: torch.optim.Optimizer, 
                 scheduler: torch.optim.lr_scheduler._LRScheduler, 
                 tasks: dict, 
                 logger: Logger, 
                 results_folder: str,
                 max_grad_norm: float = 10.0):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.tasks = tasks
        self.logger = logger
        self.results_folder = results_folder
        self.k_tasks = list(tasks.keys())
        self.n_k_tasks = len(self.k_tasks)
        if self.model.n_tasks > 1:
            train_loaders = [iter(tasks[k]["dataloaders"][0]) for k in self.k_tasks]
            assert all([len(train_loaders[0]) == len(train_loaders[i]) for i in range(self.n_k_tasks)])
            self.n_batches = len(train_loaders[0])
        else:
            self.train_loaders = [tasks[k]["dataloaders"][0] for k in self.k_tasks]
            self.n_batches = len(self.train_loaders[0])
        self.valid_loaders = [tasks[k]["dataloaders"][1] for k in self.k_tasks]
        self.test_loaders = [tasks[k]["dataloaders"][2] for k in self.k_tasks]
        
        self.loss_records = list([[], [], []] for _ in range(self.n_k_tasks))
        self.eval_records = list([[], [], [], [], []] for _ in range(self.n_k_tasks))

        self.grad_records = []
        self.max_grad_norm = max_grad_norm


    def train(self, n_epochs: int, device, verbose: bool = False, mask_mp: float = 0.0):
        """
        One batch at a time by One training
        """
        self.logger.info("training all, one batch at a time...")
        self.model.to(device)
        self.model.train()
        for epoch in range(n_epochs):
            epoch_t = time.time()
            n_ior = self.set_ior() if "IOR" in self.tasks else 0
            train_loaders = [iter(self.tasks[k]["dataloaders"][0]) for k in self.k_tasks]
            for i in range(self.n_batches):
                for j, k in enumerate(self.k_tasks):
                    class_weights = self.tasks[k].get("class_weights", None)
                    class_weights = None if class_weights is None else class_weights.to(device)
                    loss_w, loss_s = self.tasks[k]["loss_w"], self.tasks[k]["loss_s"]
                    if k == "IOR":
                        loss_1, loss_2, loss_3 = self.train_ior(n_ior, train_loaders[j], device)
                    else:
                        has_prompt = self.tasks[k].get("has_prompt", False)
                        task_id = self.tasks[k]["key"]
                        x, y, m, _, hy = next(train_loaders[j])
                        x, y, m, hy = x.to(device), y.to(device), m.to(device), hy.to(device)
                        p_m, p_y, _ = self.model(x, task_id, hy if has_prompt else None)
                        p_yy, _ = self.model.for_forward(x[:, -1])

                        loss_1 = cross_entropy(p_y[:, :, loss_s[0]], y[:, loss_s[0]], class_weights) if y.ndim > 1 else torch.tensor([0.0]).to(device)
                        loss_2 = mse_loss(p_m[:, loss_s[1]], m[:, loss_s[1]]) if m.ndim > 1 else torch.tensor([0.0]).to(device)
                        loss_3 = cross_entropy(p_yy, y[:, -1], class_weights) if y.ndim > 1 else torch.tensor([0.0]).to(device)

                    # mask loss (minimize the area of attention)
                    loss = (mask_mp * ((p_m[:, -1] + 1.0)/2.0).mean()) if mask_mp > 0.0 else 0.0
                    
                    # cross-entropy for the sequence
                    loss = loss + loss_w[0] * loss_1 if loss_w[0] > 0 else loss
                    self.loss_records[j][0].append(loss_1.item() + 1e-6)

                    # mse loss for the mask
                    loss = loss + loss_w[1] * loss_2 if loss_w[1] > 0 else loss
                    self.loss_records[j][1].append(loss_2.item() + 1e-6)
                    
                    # cross-entropy for the final prediction
                    loss = loss + loss_w[2] * loss_3 if loss_w[2] > 0 else loss
                    self.loss_records[j][2].append(loss_3.item() + 1e-6)

                    # # grad backprop, clip and update
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    self.grad_records.append(get_grad_norms(self.model))
                    clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
            # save the model and optimizer
            torch.save(self.model.state_dict(), os.path.join(self.results_folder, f"model_{epoch}_" + ".pth"))
            torch.save(self.optimizer.state_dict(), os.path.join(self.results_folder, f"optimizer_{epoch}_" + ".pth"))
            # update the scheduler
            self.scheduler.step()

            # log and plot
            self.logger.info(f"Epoch {epoch+1}/{n_epochs} ({time.time() - epoch_t:.2f}s):")
            for j, k in enumerate(self.k_tasks):
                self.logger.info(f"  Task {k}:")
                self.logger.info(f"    Loss {0}: {sum(self.loss_records[j][0][-self.n_batches:])/self.n_batches:.6f}"
                                    f"    Loss {1}: {sum(self.loss_records[j][1][-self.n_batches:])/self.n_batches:.6f}"
                                    f"    Loss {2}: {sum(self.loss_records[j][2][-self.n_batches:])/self.n_batches:.6f}")
            if verbose:
                plot_all(10, self.model, self.tasks, self.results_folder, f"_ep_{epoch+1}", device, self.logger, False)
                self.eval(device, track=True)

        self.undo_ior()
        self.model.eval()

    def set_ior(self):
        if "IOR" in self.tasks:
            n_digits = self.tasks["IOR"]["params"]["n_digits"]
            n_attend = self.tasks["IOR"]["params"]["n_attend"]
            rand_n = torch.randint(1, n_digits + 1, (1, )).item()
            self.tasks["IOR"]["dataloaders"][0].dataset.n_digits = rand_n
            self.tasks["IOR"]["dataloaders"][0].dataset.n_iter = rand_n * n_attend
            return rand_n
        return 0

    def undo_ior(self, ):
        if "IOR" in self.tasks:
            n_digits = self.tasks["IOR"]["params"]["n_digits"]
            n_attend = self.tasks["IOR"]["params"]["n_attend"]
            self.tasks["IOR"]["dataloaders"][0].dataset.n_digits = n_digits
            self.tasks["IOR"]["dataloaders"][0].dataset.n_iter = n_digits * n_attend

    def train_ior(self, n_: int, dloader_, device_):
        task_id = self.tasks["IOR"]["key"]
        has_prompt = self.tasks["IOR"].get("has_prompt", False)
        n_attend = self.tasks["IOR"]["params"]["n_attend"]
        # rand_n_digits = torch.randint(2, n_digits + 1, (1, )).item()
        x, y, m, _, hy = next(dloader_)
        x, y, m, hy = x.to(device_), y.to(device_), m.to(device_), hy.to(device_)
        p_m, p_y, a_ = self.model(x, task_id, hy if has_prompt else None)
        m = 0.5 * (m + 1.0)
        p_m = 0.5 * (p_m + 1.0)
        with torch.no_grad():
            target_ids = []
            targets_masks = []
            targets_labels = []
            batch_ids = torch.arange(x.size(0)).to(device_)
            for i in range(n_):
                j = (i * n_attend + n_attend - 1, )
                target_ids.append(get_ior_match(p_m[:, j], m))
                targets_masks.append(m[batch_ids, target_ids[-1]])
                targets_labels.append(y[batch_ids, target_ids[-1]])
                m[batch_ids, target_ids[-1]] = -1.0

        loss_1 = 0.0
        loss_2 = 0.0
        loss_3 = torch.tensor([0.0]).to(device_)
        for i in range(n_):
            j = i * n_attend + n_attend - 1
            loss_1 = loss_1 + cross_entropy(p_y[:, :, j], targets_labels[i])
            loss_2 = loss_2 + mse_loss(p_m[:, j], targets_masks[i])
        return loss_1, loss_2, loss_3

    def eval(self, device, valid = True, track = False):
        _loader = self.valid_loaders if valid else self.test_loaders
        self.logger.info("validating..." if valid else "testing...")
        eval_scores = list([0.0, 0.0, 0.0, 0.0, 0] for _ in range(self.n_k_tasks))
        
        self.model.to(device)
        self.model.eval()
        with torch.no_grad():
            for j, k in enumerate(self.k_tasks):
                loss_w, loss_s = self.tasks[k]["loss_w"], self.tasks[k]["loss_s"]
                if k == "IOR":
                    eval_scores[j] = self.eval_ior(_loader[j], device)
                else:
                    has_prompt = self.tasks[k].get("has_prompt", False)
                    task_id = self.tasks[k]["key"]
                    class_weights = self.tasks[k].get("class_weights", None)
                    class_weights = None if class_weights is None else class_weights.to(device)
                    for x, y, m, _, hy in _loader[j]:
                        x, y, m, hy = x.to(device), y.to(device), m.to(device), hy.to(device)
                        p_m, p_y, a_ = self.model(x, task_id, hy if has_prompt else None)
                        p_yy, aa_ = self.model.for_forward(x[:, -1])
                        prediction = p_yy.argmax(dim=1, keepdim=True)
                        eval_scores[j][0] += cross_entropy(p_y[:, :, loss_s[0]], y[:, loss_s[0]], class_weights, reduction='sum').item() if y.ndim > 1 else 0.0
                        eval_scores[j][1] += cross_entropy(p_yy, y[:, -1], class_weights, reduction='sum').item() if y.ndim > 1 else 0.0
                        eval_scores[j][2] += pixel_error(p_m[:, loss_s[1]], m[:, loss_s[1]]).item() * x.size(0) if m.ndim > 1 else 0.0
                        eval_scores[j][3] += normed_acc(p_m[:, loss_s[1]], m[:, loss_s[1]]).item() * x.size(0) if m.ndim > 1 else 0.0
                        eval_scores[j][4] += prediction.eq(y[:, -1].view_as(prediction)).sum().item() if y.ndim > 1 else 0

                self.logger.info(f"  Task {k}:")
                n_samples = _loader[j].dataset.__len__()
                eval_scores[j][0] /= n_samples
                eval_scores[j][1] /= n_samples
                eval_scores[j][2] /= n_samples
                eval_scores[j][3] /= n_samples
                self.logger.info(f"    CEi Loss: {eval_scores[j][0]:.6f}"
                                 f"    CEe Loss: {eval_scores[j][1]:.6f}"
                                 f"    Pix Err: {eval_scores[j][2]:.6f}"
                                 f"    Att Acc: {eval_scores[j][3]:.6f}"
                                 f"    Cls Acc: {eval_scores[j][4]}/{n_samples}")
                if track:
                    self.eval_records[j][0].append(eval_scores[j][0])
                    self.eval_records[j][1].append(eval_scores[j][1])
                    self.eval_records[j][2].append(eval_scores[j][2])
                    self.eval_records[j][3].append(eval_scores[j][3])
                    self.eval_records[j][4].append(eval_scores[j][4]/n_samples)
        
    def eval_ior(self, dloader_, device_):
        eval_scores_ = [0.0, 0.0, 0.0, 0.0, 0]
        task_id = self.tasks["IOR"]["key"]
        has_prompt = self.tasks["IOR"].get("has_prompt", False)
        n_attend = self.tasks["IOR"]["params"]["n_attend"]
        n_digits = self.tasks["IOR"]["params"]["n_digits"]
        for x, y, m, _, hy in dloader_:
            x, y, m, hy = x.to(device_), y.to(device_), m.to(device_), hy.to(device_)
            p_m, p_y, a_ = self.model(x, task_id, hy if has_prompt else None)
            m = 0.5 * (m + 1.0)
            p_m = 0.5 * (p_m + 1.0)
            target_ids = []
            batch_ids = torch.arange(x.size(0)).to(device_)
            for i in range(n_digits):
                j = (i * n_attend + n_attend - 1, )
                target_ids = get_ior_match(p_m[:, j], m)

                prediction = p_y[:, :, j[0]].argmax(dim=1, keepdim=True)
                eval_scores_[0] += cross_entropy(p_y[:, :, j[0]], y[batch_ids, target_ids], reduction='sum').item() / n_digits
                eval_scores_[2] += pixel_error(p_m[:, j[0]], m[batch_ids, target_ids], donorm=False).item() * x.size(0) / n_digits
                eval_scores_[3] += normed_acc(p_m[:, j[0]], m[batch_ids, target_ids], donorm=False).item() * x.size(0) / n_digits
                eval_scores_[4] += prediction.eq(y[batch_ids, target_ids].view_as(prediction)).sum().item()

                m[batch_ids, target_ids] = -1.0
        eval_scores_[4] = int(eval_scores_[4] / n_digits)
        return eval_scores_


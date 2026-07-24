from models.lora_utils.ema_state import EMAStats
import torch, numpy as np
from torch.nn import functional as F
from utils.args import *
from models.utils.continual_model import ContinualModel
from models.lora_utils.lora import LoRA_ViT_timm
import timm

class CLoRA(ContinualModel):
    NAME = 'clora'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        parser = ArgumentParser(description='Continual learning via lora.')
        parser.add_argument('--loss_window_length', default=5, type=int, help='number of prompts (M in paper)')
        parser.add_argument('--n_classes', default=100, type=int, help='number of prompts (M in paper)')
        parser.add_argument('--alpha', default=2, type=float, help='number of prompts (M in paper)')
        parser.add_argument('--beta', default=1, type=float, help='number of prompts (M in paper)')
        parser.add_argument('--gama', default=1, type=float, help='number of prompts (M in paper)')

        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):

        del backbone
        model = timm.create_model("vit_base_patch16_224", pretrained=True)
        self.numclass = args.n_classes
        backbone = LoRA_ViT_timm(vit_model=model, r=4, num_classes=self.numclass)
        super().__init__(backbone, loss, args, transform, dataset=dataset)
        # ===== Loss window for task boundary detection =====
        self.loss_window, self.loss_window_length = [], args.loss_window_length
        self.last_loss_window_mean, self.last_loss_window_variance, self.new_peak_detected = 0.0, 0.0, True

        # ===== MAS-style LoRA importance =====
        self.count_updates, self.omega_As, self.omega_Bs = 0, [], []


        # ===== Online class statistics (EMA) =====
        self.feature_dim = 768
        self._class_means = torch.zeros(self.numclass, self.feature_dim)
        self._class_covs = torch.zeros(self.numclass, self.feature_dim, self.feature_dim)
        self._class_count = torch.zeros(self.numclass)

        self.ema_stats = EMAStats(num_classes=self.numclass, feat_dim=768, alpha=0.1, device=self.device)

    # ------------------------------------------------------------------
    # Forward utilities
    # ------------------------------------------------------------------
    def _forward_pass(self, inputs, model):
        features, outputs = model.forward(inputs, return_features=True)
        cls_token, patch_tokens = features[:, 0], features[:, 1:]
        return features, outputs, cls_token, patch_tokens

    # ------------------------------------------------------------------
    # Loss components
    # ------------------------------------------------------------------
    def _compute_primary_loss(self, outputs, labels):
        ce_loss = self.loss(outputs, labels)
        # print(f'CE loss: {ce_loss}')
        return ce_loss

    def _prototype_invariance_loss(self):
        """
        约束旧类 prototype 的预测保持不变
        """
        if not hasattr(self, '_class_means'):
            return 0.0

        proto = self._class_means.to(self.device)  # [C, D]
        valid = torch.nonzero(proto.abs().sum(dim=1) != 0).squeeze()

        if valid.numel() == 0:
            return 0.0

        proto = proto[valid]
        labels = valid.to(self.device)

        logits = self.net.lora_vit.head(proto)[:, :self.numclass]
        invariance_loss = F.cross_entropy(logits, labels)
        # print(f'Invariance loss: {invariance_loss}')
        return invariance_loss

    # ------------------------------------------------------------------
    # prototype semantic augmentation loss
    # ------------------------------------------------------------------
    def _prototype_semantic_augmentation_loss(self):
        if self._class_means is None: return 0.0
        valid_classes = torch.nonzero(self._class_means.abs().sum(dim=1) != 0).squeeze().tolist()
        if isinstance(valid_classes, int): valid_classes = [valid_classes]
        proto, proto_labels = [], []
        for _ in range(self.args.batch_size):
            idx = np.random.choice(valid_classes); proto.append(self._class_means[idx]); proto_labels.append(idx)
        proto = torch.stack(proto).to(self.device); proto_labels = torch.tensor(proto_labels).to(self.device)
        logits = self.net.lora_vit.head(proto)[:, :self.numclass]
        aug_logits = self._seman_aug(proto, logits, proto_labels, ratio=0.1)
        semanAug_loss = F.cross_entropy(aug_logits, proto_labels)
        # print(f'SemAug loss: {semanAug_loss}')
        return semanAug_loss

    def _seman_aug(self, features, logits, labels, ratio):
        W = self.net.lora_vit.head.weight[:self.numclass]; W_y = W[labels]
        W_diff = W.unsqueeze(0) - W_y.unsqueeze(1)
        Sigma_y = self._class_covs[labels].to(features.device)
        sigma_diag = torch.clamp(torch.diagonal(Sigma_y, dim1=1, dim2=2), min=1e-6, max=0.1)
        var_term = torch.clamp(torch.sum(W_diff ** 2 * sigma_diag.unsqueeze(1), dim=2), max=50.0)
        return logits + 0.5 * ratio * var_term

    def _update_lora_importance(self):
        eps = 1e-8

        # ============================================================
        # 0. step 计数
        # ============================================================
        if not hasattr(self, 'global_step'):
            self.global_step = 0
            self.steps_since_reset = 0
            self.last_fast_step = -1e9
            self.fast_trigger_count = 0

            # FAST trigger 超参数（推荐默认）
            self.fast_warmup_steps = 30
            self.fast_cooldown_steps = 50
            self.fast_trigger_patience = 3

        self.global_step += 1
        self.steps_since_reset += 1

        # ============================================================
        # 1. 收集 LoRA 梯度向量 & 梯度能量
        # ============================================================
        grad_vec = []
        for p in self.net.parameters():
            if (getattr(p, '_is_wnew_a', False) or getattr(p, '_is_wnew_b', False)) and p.grad is not None:
                grad_vec.append(p.grad.detach().flatten())

        if len(grad_vec) == 0:
            return

        grad_vec = torch.cat(grad_vec)
        grad_energy = torch.norm(grad_vec).item()

        # ============================================================
        # 2. FAST TIME SCALE —— 新任务瞬时检测
        # ============================================================
        fast_signal = False

        # (a) 梯度能量跳变
        if hasattr(self, 'last_grad_energy'):
            jump_ratio = grad_energy / (self.last_grad_energy + eps)
            if jump_ratio > 2.0:
                fast_signal = True
                # print('>>> FAST signal: gradient energy jump')

        # (b) 梯度方向冲突
        grad_conflict = 0.0
        if hasattr(self, 'old_grad_vec') and self.old_grad_vec is not None:
            cos_sim = torch.dot(grad_vec, self.old_grad_vec) / (
                    torch.norm(grad_vec) * torch.norm(self.old_grad_vec) + eps
            )
            grad_conflict = (1 - cos_sim).item()
            if grad_conflict > 0.5:
                fast_signal = True
                # print('>>> FAST signal: gradient conflict')

        self.last_grad_energy = grad_energy
        self.old_grad_vec = grad_vec.clone().detach()

        # ============================================================
        # 3. FAST trigger 去抖动 / 冷却 / warmup
        # ============================================================
        can_fast = (
                self.steps_since_reset > self.fast_warmup_steps and
                self.global_step - self.last_fast_step > self.fast_cooldown_steps
        )

        if fast_signal and can_fast:
            self.fast_trigger_count += 1
        else:
            self.fast_trigger_count = max(0, self.fast_trigger_count - 1)

        if self.fast_trigger_count >= self.fast_trigger_patience:
            self.count_updates += 1
            # print(f'---------------- FAST LoRA Expansion: {self.count_updates} ----------------')

            self.net.update_and_reset_lora_parameters()
            self.net = self.net.to(self.device)
            self._old_class_covs = (
                self._class_covs.clone().detach()
                if self._class_covs is not None else None
            )

            # reset FAST states
            self.last_fast_step = self.global_step
            self.steps_since_reset = 0
            self.fast_trigger_count = 0
            self.new_peak_detected = True

            # reset slow window
            self.grad_energy_window = []

            return

        # ============================================================
        # 4. SLOW TIME SCALE —— 梯度能量稳定性
        # ============================================================
        if not hasattr(self, 'grad_energy_window'):
            self.grad_energy_window = []

        self.grad_energy_window.append(grad_energy)
        if len(self.grad_energy_window) > self.loss_window_length:
            self.grad_energy_window.pop(0)

        if len(self.grad_energy_window) < self.loss_window_length:
            return

        energy_arr = np.array(self.grad_energy_window)
        energy_mean = energy_arr.mean()
        energy_var = energy_arr.var()
        if not hasattr(self, 'last_energy_mean'):
            self.last_energy_mean = energy_mean
            self.last_energy_var = energy_var
            return

        x = np.arange(len(energy_arr))
        slope, _ = np.polyfit(x, energy_arr, 1)
        slope = abs(slope)

        mean_rel = abs(energy_mean - getattr(self, 'last_energy_mean', energy_mean)) / (
                abs(getattr(self, 'last_energy_mean', energy_mean)) + eps
        )
        var_rel = abs(energy_var - getattr(self, 'last_energy_var', energy_var)) / (
                abs(getattr(self, 'last_energy_var', energy_var)) + eps
        )

        # print(f'GradEnergy mean: {energy_mean}, var: {energy_var}, slope: {slope}')
        # print(f'Mean rel: {mean_rel}, Var rel: {var_rel}')

        energy_stable = slope < (mean_rel + var_rel + eps)
        # print(f'Grad conflict: {grad_conflict}')
        conflict_low = grad_conflict < 0.4


        # ============================================================
        # 5. SLOW trigger —— 重要性估计 + 冻结
        # ============================================================
        if energy_stable and conflict_low and self.new_peak_detected:
            self.count_updates += 1
            # print(f'---------------- SLOW Importance Update: {self.count_updates} ----------------')

            self.last_energy_mean = energy_mean
            self.last_energy_var = energy_var
            self.new_peak_detected = False

            # -------- 计算 LoRA 参数重要性（梯度平方）--------
            gradients_A = [0 for p in self.net.parameters() if getattr(p, '_is_wnew_a', False)]
            gradients_B = [0 for p in self.net.parameters() if getattr(p, '_is_wnew_b', False)]

            wnew_a = filter(lambda p: getattr(p, '_is_wnew_a', False), self.net.parameters())
            wnew_b = filter(lambda p: getattr(p, '_is_wnew_b', False), self.net.parameters())

            for idx, (p_a, p_b) in enumerate(zip(wnew_a, wnew_b)):
                gradients_A[idx] += (p_a.grad.detach().cpu().numpy()) ** 2
                gradients_B[idx] += (p_b.grad.detach().cpu().numpy()) ** 2

            omega_As_old = self.omega_As[:]
            omega_Bs_old = self.omega_Bs[:]
            self.omega_As, self.omega_Bs = [], []

            for idx, (gA, gB) in enumerate(zip(gradients_A, gradients_B)):
                if len(omega_As_old) != 0:
                    self.omega_As.append(
                        (1 / self.count_updates) * gA +
                        (1 - 1 / self.count_updates) * omega_As_old[idx]
                    )
                    self.omega_Bs.append(
                        (1 / self.count_updates) * gB +
                        (1 - 1 / self.count_updates) * omega_Bs_old[idx]
                    )
                else:
                    self.omega_As.append(gA)
                    self.omega_Bs.append(gB)

            self.net.update_and_reset_lora_parameters()
            self.net = self.net.to(self.device)
            self._cache_old_lora_subspace()
            self._old_class_covs = self._class_covs.clone().detach()
            self.last_grad_conflict = grad_conflict

    def _orthogonalize_lora_gradients(self):
        """
        对新 LoRA 梯度做子空间正交
        """
        if not hasattr(self, 'old_lora_subspace'):
            return
        for p in self.net.parameters():
            if (getattr(p, '_is_wnew_a', False) or getattr(p, '_is_wnew_b', False)) \
                    and p.grad is not None:
                p_vec = p.detach().flatten().unsqueeze(1)  # [dr, 1]
                g_vec = p.grad.detach().flatten().unsqueeze(1)  # [dr, 1]

                norm = torch.norm(p_vec) + 1e-8
                p_unit = p_vec / norm

                proj = p_unit @ (p_unit.T @ g_vec)  # [dr, 1]

                conflict = getattr(self, 'last_grad_conflict', 0.0)
                conflict = torch.as_tensor(conflict, device=g_vec.device).detach()
                alpha = torch.exp(-0.001 * conflict)

                g_new = g_vec - proj + alpha * proj

                p.grad.copy_(g_new.view_as(p.grad))


    def _cache_old_lora_subspace(self, rank_k=8):
        """
        保存旧 LoRA 子空间，用于正交约束
        """
        self.old_lora_subspace = []

        for p in self.net.parameters():
            if getattr(p, '_is_wnew_a', False) or getattr(p, '_is_wnew_b', False):
                W = p.data.detach().view(p.data.size(0), -1)  # [out, in]
                try:
                    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
                    basis = Vh[:rank_k].detach()  # [k, in]
                    self.old_lora_subspace.append(basis)
                except:
                    self.old_lora_subspace.append(None)

    def ncm_logits(self, features):
        """
        features: [B, D]
        return logits: [B, C]
        """
        class_means = self._class_means.to(features.device)

        # mask 未初始化的类
        valid = (class_means.abs().sum(dim=1) != 0)

        # L2 normalize
        features = F.normalize(features, dim=1)
        class_means = F.normalize(class_means, dim=1)

        logits = features @ class_means.T  # cosine similarity

        # invalid class 给极小值
        logits[:, ~valid] = -1e9
        return logits

    # ------------------------------------------------------------------
    # Main training interface
    # ------------------------------------------------------------------
    def observe(self, inputs, labels, not_aug_inputs):
        self.opt.zero_grad()
        features, outputs, cls_token, patch_tokens = self._forward_pass(inputs, self.net)

        # update online class statistics
        self.ema_stats.update(cls_token, labels)
        self._class_means, self._class_covs = self.ema_stats.get_stats()
        # 使用 NCM 代替 head
        logits_ncm = self.ncm_logits(cls_token)
        logits_head = self.net.lora_vit.head(cls_token)[:, :self.numclass]
        if self.new_peak_detected:
            lambda_head = 1.2
        else:
            lambda_head = 0.2
        logits = logits_ncm + lambda_head * logits_head

        _, pred = torch.max(logits, 1)
        loss = self.args.alpha * self._compute_primary_loss(outputs, labels)


        loss += self.args.beta * self._prototype_semantic_augmentation_loss()
        # 因果 prototype 稳定
        loss += self.args.gama * self._prototype_invariance_loss()

        loss.backward()

        if not self.new_peak_detected:
            self._orthogonalize_lora_gradients()

        self.opt.step()
        self._update_lora_importance()


        return loss.item()
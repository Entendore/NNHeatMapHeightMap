import sys
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtGui import QPalette, QColor
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QPushButton, QComboBox, QDoubleSpinBox, 
                               QLabel, QSpinBox, QStatusBar, QGroupBox, QFormLayout,
                               QCheckBox, QFileDialog, QLineEdit, QTabWidget, QSlider,
                               QSizePolicy)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt

plt.style.use('dark_background')

# ------------------------------------------
# 1. Loss Functions & Blocks
# ------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce)
        loss = self.alpha * (1 - pt)**self.gamma * bce
        return loss.mean()

class ResidualBlock(nn.Module):
    def __init__(self, dim, act, dropout, use_bn):
        super().__init__()
        layers = [nn.Linear(dim, dim), act()]
        if use_bn: layers.append(nn.BatchNorm1d(dim))
        if dropout > 0: layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dim, dim))
        if use_bn: layers.append(nn.BatchNorm1d(dim))
        self.block = nn.Sequential(*layers)
        self.act = act()
    def forward(self, x):
        return self.act(x + self.block(x))

# ------------------------------------------
# 2. Modern Architecture Definitions
# ------------------------------------------
class MLPPreset(nn.Module):
    def __init__(self, input_dim, custom_str, fallback_size, fallback_layers, fallback_act, dropout=0.0, use_bn=False, use_res=False):
        super().__init__()
        act_map = {"Tanh": nn.Tanh, "ReLU": nn.ReLU, "LeakyReLU": nn.LeakyReLU, "ELU": nn.ELU, "Sigmoid": nn.Sigmoid}
        layers = []
        last_dim = input_dim
        if custom_str:
            tokens = [t.strip() for t in custom_str.split(',')]
            for token in tokens:
                if token.isdigit():
                    dim = int(token)
                    layers.append(nn.Linear(last_dim, dim))
                    last_dim = dim
                else:
                    act_class = act_map.get(token)
                    if act_class:
                        layers.append(act_class())
                        if use_bn: layers.append(nn.BatchNorm1d(last_dim))
                        if dropout > 0: layers.append(nn.Dropout(dropout))
        else:
            act_class = act_map.get(fallback_act, nn.Tanh)
            for _ in range(fallback_layers):
                layers.append(nn.Linear(last_dim, fallback_size))
                last_dim = fallback_size
                if use_res and last_dim == fallback_size:
                    layers.append(ResidualBlock(last_dim, act_class, dropout, use_bn))
                else:
                    layers.append(act_class())
                    if use_bn: layers.append(nn.BatchNorm1d(last_dim))
                    if dropout > 0: layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(last_dim, 1))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

class TransformerPreset(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = 16
        self.embedding = nn.Linear(1, self.d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, 2, self.d_model))
        layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=2, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Linear(self.d_model, 1)
    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.embedding(x) + self.pos_encoder
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.head(x)

class MambaPreset(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = 16
        self.embedding = nn.Linear(1, self.d_model)
        self.proj_in = nn.Linear(self.d_model, self.d_model * 2)
        self.proj_out = nn.Linear(self.d_model, self.d_model)
        self.A = nn.Parameter(torch.randn(self.d_model)) 
        self.head = nn.Linear(self.d_model, 1)
    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.embedding(x) 
        batch_size, seq_len, _ = x.shape
        h = torch.zeros(batch_size, self.d_model, device=x.device)
        outputs = []
        A_bar = torch.exp(-torch.sigmoid(self.A)) 
        for t in range(seq_len):
            h = A_bar * h + x[:, t, :]
            outputs.append(h)
        x = torch.stack(outputs, dim=1)
        x = self.proj_in(x)
        gate, x = x.chunk(2, dim=-1)
        x = x * torch.sigmoid(gate)
        x = self.proj_out(x)
        x = x.mean(dim=1)
        return self.head(x)

class AutoencoderPreset(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 2))
        self.decoder = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 2))
        self.classifier = nn.Linear(2, 1)
    def forward(self, x):
        latent = self.encoder(x)
        recon = self.decoder(latent)
        logit = self.classifier(latent)
        return logit, recon, latent

# ------------------------------------------
# 3. Feature Engineering Utility
# ------------------------------------------
def expand_features(X, mode):
    if mode == "Linear": return X
    elif mode == "Poly2": return np.hstack([X, X**2])
    elif mode == "Poly3": return np.hstack([X, X**2, (X[:, 0] * X[:, 1]).reshape(-1, 1)])

# ------------------------------------------
# 4. The Neural Network & Training Worker
# ------------------------------------------
class TrainingWorker(QThread):
    update_plots = Signal(dict)
    finished_training = Signal(str)

    def __init__(self, config):
        super().__init__()
        self._is_running = True
        self._is_paused = False
        self.config = config
        torch.manual_seed(config['seed'])
        np.random.seed(config['seed'])
        self.model = self._build_model()
        self._init_weights()
        self.generate_dataset()

    def _build_model(self):
        c = self.config
        if c['architecture'] == "MLP":
            return MLPPreset(c['input_dim'], c['custom_mlp'], c['mlp_hidden'], c['mlp_layers'], c['mlp_activation'], c['dropout'], c['use_bn'], c['use_res'])
        elif c['architecture'] == "Transformer": return TransformerPreset()
        elif c['architecture'] == "Mamba": return MambaPreset()
        elif c['architecture'] == "Autoencoder": return AutoencoderPreset()

    def _init_weights(self):
        m = self.config['init_method']
        def init_fn(layer):
            if isinstance(layer, nn.Linear):
                if m == "Xavier": nn.init.xavier_uniform_(layer.weight)
                elif m == "Kaiming": nn.init.kaiming_uniform_(layer.weight)
                else: nn.init.uniform_(layer.weight, -0.1, 0.1)
                if layer.bias is not None: nn.init.zeros_(layer.bias)
        self.model.apply(init_fn)

    def generate_dataset(self):
        N = 500
        noise = self.config['noise']
        ds_type = self.config['dataset_type']
        ratio = self.config['imbalance_ratio']
        n0 = int(N * ratio)
        n1 = N - n0
        
        if ds_type == "Circles":
            a0, a1 = np.random.rand(n0)*2*np.pi, np.random.rand(n1)*2*np.pi
            X0 = np.c_[np.cos(a0)*1.0, np.sin(a0)*1.0] + np.random.randn(n0, 2)*noise*0.5
            X1 = np.c_[np.cos(a1)*3.0, np.sin(a1)*3.0] + np.random.randn(n1, 2)*noise*0.5
            X, y = np.vstack([X0, X1]), np.hstack([np.zeros(n0), np.ones(n1)])
        elif ds_type == "Moons":
            a1, a2 = np.linspace(0, np.pi, n0), np.linspace(np.pi, 2*np.pi, n1)
            X1 = np.c_[np.cos(a1)*2, np.sin(a1)*2] + np.random.randn(n0, 2)*noise
            X2 = np.c_[1+np.cos(a2)*2, -0.5+np.sin(a2)*2] + np.random.randn(n1, 2)*noise
            X, y = np.vstack([X1, X2]), np.hstack([np.zeros(n0), np.ones(n1)])
        elif ds_type == "XOR":
            n0_q, n1_q = n0//2, n1//2
            X = np.vstack([np.random.randn(n0_q,2)*noise+[2,2], np.random.randn(n0_q,2)*noise+[-2,-2],
                           np.random.randn(n1_q,2)*noise+[2,-2], np.random.randn(n1-n1_q,2)*noise+[-2,2]])
            y = np.hstack([np.zeros(n0), np.ones(n1)])
        elif ds_type == "Spirals":
            t0, t1 = np.sqrt(np.random.rand(n0))*3*np.pi, np.sqrt(np.random.rand(n1))*3*np.pi
            c1 = np.c_[np.cos(t0)*t0*0.5, np.sin(t0)*t0*0.5] + np.random.randn(n0, 2)*noise
            c2 = np.c_[np.cos(t1+np.pi)*t1*0.5, np.sin(t1+np.pi)*t1*0.5] + np.random.randn(n1, 2)*noise
            X, y = np.vstack([c1, c2]), np.hstack([np.zeros(n0), np.ones(n1)])
            
        self.X_orig = X.copy()
        X_expanded = expand_features(X, self.config['feature_mode'])
        self.config['input_dim'] = X_expanded.shape[1]
        self.model = self._build_model()
        self._init_weights()
        
        self.xs = np.linspace(-5, 5, 50)
        self.ys = np.linspace(-5, 5, 50)
        self.X_grid, self.Y_grid = np.meshgrid(self.xs, self.ys)
        grid_expanded = expand_features(np.c_[self.X_grid.ravel(), self.Y_grid.ravel()], self.config['feature_mode'])
        self.grid_tensor = torch.FloatTensor(grid_expanded)
        
        indices = np.random.permutation(N)
        train_idx, val_idx = indices[:int(0.8*N)], indices[int(0.8*N):]
        self.X_train = torch.FloatTensor(X_expanded[train_idx])
        self.y_train = torch.FloatTensor(y[train_idx].reshape(-1, 1))
        self.X_val = torch.FloatTensor(X_expanded[val_idx])
        self.y_val = torch.FloatTensor(y[val_idx].reshape(-1, 1))
        self.X = torch.FloatTensor(X_expanded)
        self.y = torch.FloatTensor(y.reshape(-1, 1))
        self.train_mask = torch.zeros(N, dtype=torch.bool)
        self.train_mask[train_idx] = True

    def _get_weights_and_grads(self):
        w1 = np.array([])
        if self.config['architecture'] == "MLP": w1 = self.model.net[0].weight.detach().numpy().flatten()
        elif self.config['architecture'] == "Transformer": w1 = self.model.embedding.weight.detach().numpy().flatten()
        elif self.config['architecture'] == "Mamba": w1 = self.model.embedding.weight.detach().numpy().flatten()
        elif self.config['architecture'] == "Autoencoder": w1 = self.model.encoder[0].weight.detach().numpy().flatten()
        grad_norms = [p.grad.detach().norm(2).item() for p in self.model.parameters() if p.grad is not None]
        return w1, grad_norms

    def run(self):
        c = self.config
        if c['loss_func'] == "Hinge":
            criterion = lambda logits, targets: torch.mean(torch.clamp(1 - (2*targets - 1) * logits, min=0))
        elif c['loss_func'] == "Focal": criterion = FocalLoss(alpha=0.25, gamma=2.0)
        else: criterion = nn.BCEWithLogitsLoss()
        mse_loss = nn.MSELoss()
        
        opts = {"Adam": torch.optim.Adam, "SGD": torch.optim.SGD, "RMSprop": torch.optim.RMSprop}
        optimizer = opts[c['optimizer_name']](self.model.parameters(), lr=c['lr'], weight_decay=c['weight_decay'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200) if c['use_scheduler'] else None
        
        loss_hist, val_loss_hist, acc_hist, val_acc_hist = [], [], [], []
        N = self.X_train.shape[0]
        epoch = 0
        best_val_loss, patience_counter = float('inf'), 0
        stop_reason = "Manual Stop"
        
        while self._is_running:
            while self._is_paused and self._is_running: self.msleep(100)
            if not self._is_running: break

            self.model.train()
            indices = torch.randperm(N)
            epoch_loss, epoch_correct = 0.0, 0.0
            
            for i in range(0, N, c['batch_size']):
                if not self._is_running or self._is_paused: break
                batch_idx = indices[i:i+c['batch_size']]
                X_batch, y_batch = self.X_train[batch_idx], self.y_train[batch_idx]
                if c['aug_noise'] > 0: X_batch = X_batch + torch.randn_like(X_batch) * c['aug_noise']
                
                optimizer.zero_grad()
                if c['architecture'] == "Autoencoder":
                    logit, recon, _ = self.model(X_batch)
                    loss = criterion(logit, y_batch) + 0.1 * mse_loss(recon, X_batch)
                else:
                    out = self.model(X_batch)
                    loss = criterion(out, y_batch)
                loss.backward()
                if c['grad_clip'] > 0: torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=c['grad_clip'])
                optimizer.step()

                epoch_loss += loss.item()
                preds = (out > 0).float() if c['loss_func'] == "Hinge" else (torch.sigmoid(out) > 0.5).float()
                epoch_correct += (preds == y_batch).float().sum().item()

            avg_loss, avg_acc = epoch_loss / (N / c['batch_size']), epoch_correct / N
            
            self.model.eval()
            with torch.no_grad():
                if c['architecture'] == "Autoencoder":
                    val_logit, val_recon, _ = self.model(self.X_val)
                    val_loss = criterion(val_logit, self.y_val) + 0.1 * mse_loss(val_recon, self.X_val)
                else:
                    val_out = self.model(self.X_val)
                    val_loss = criterion(val_out, self.y_val)
                val_preds = (val_out > 0).float() if c['loss_func'] == "Hinge" else (torch.sigmoid(val_out) > 0.5).float()
                val_acc = (val_preds == self.y_val).float().mean().item()

            if scheduler and not self._is_paused: scheduler.step()

            loss_hist.append(avg_loss); acc_hist.append(avg_acc)
            val_loss_hist.append(val_loss.item()); val_acc_hist.append(val_acc)
            if len(loss_hist) > 2000:
                loss_hist, val_loss_hist, acc_hist, val_acc_hist = loss_hist[-2000:], val_loss_hist[-2000:], acc_hist[-2000:], val_acc_hist[-2000:]
                
            if c['early_stopping'] > 0:
                if val_loss.item() < best_val_loss: best_val_loss, patience_counter = val_loss.item(), 0
                else:
                    patience_counter += 1
                    if patience_counter >= c['early_stopping']:
                        stop_reason = f"Early Stopped (Patience: {c['early_stopping']})"
                        self._is_running = False

            if epoch % c['update_interval'] == 0 or not self._is_running:
                with torch.no_grad():
                    if c['architecture'] == "Autoencoder":
                        full_logits, _, _ = self.model(self.X)
                        Z = self.model(self.grid_tensor)[0].numpy().reshape(self.X_grid.shape)
                    else:
                        full_logits = self.model(self.X)
                        Z = self.model(self.grid_tensor).numpy().reshape(self.X_grid.shape)
                    full_preds = (full_logits > 0).float() if c['loss_func'] == "Hinge" else (torch.sigmoid(full_logits) > 0.5).float()
                    misclassified_mask = (full_preds != self.y).squeeze()
                
                w1, grad_norms = self._get_weights_and_grads()
                latent_np, latent_y = None, None
                if c['architecture'] == "Autoencoder":
                    _, _, latent = self.model(self.X_val)
                    latent_np = latent.numpy()
                    latent_y = self.y_val.numpy().flatten()
                
                self.update_plots.emit({
                    'epoch': epoch, 'train_loss': avg_loss, 'train_acc': avg_acc,
                    'val_loss': val_loss.item(), 'val_acc': val_acc, 'lr': optimizer.param_groups[0]['lr'],
                    'Z': Z, 'loss_hist': loss_hist.copy(), 'val_loss_hist': val_loss_hist.copy(),
                    'acc_hist': acc_hist.copy(), 'val_acc_hist': val_acc_hist.copy(),
                    'w1': w1, 'grads': grad_norms, 'latent': latent_np, 'latent_y': latent_y,
                    'misclassified_mask': misclassified_mask.numpy(),
                    'data_logits': full_logits.numpy().flatten()
                })
            epoch += 1
        self.finished_training.emit(stop_reason)

    def stop(self): self._is_running, self._is_paused = False, False
    def set_paused(self, paused): self._is_paused = paused

# ------------------------------------------
# 5. The Main Application Window
# ------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pro Deep Learning Visualizer v10.0 - Refined UI")
        self.resize(1400, 900)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # --- Left Control Panel ---
        left_panel = QWidget()
        left_panel.setFixedWidth(330)
        left_panel.setStyleSheet("QWidget { background-color: #252525; }")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(10)
        
        # 1. Persistent Execution Bar
        exec_grp = QGroupBox("Execution & Status")
        exec_layout = QVBoxLayout()
        
        btn_row1 = QHBoxLayout()
        self.btn_start = QPushButton("Start"); self.btn_start.clicked.connect(self.start_training)
        self.btn_pause = QPushButton("Pause"); self.btn_pause.setEnabled(False); self.btn_pause.clicked.connect(self.toggle_pause)
        self.btn_stop = QPushButton("Stop"); self.btn_stop.setEnabled(False); self.btn_stop.clicked.connect(self.stop_training)
        for btn in [self.btn_start, self.btn_pause, self.btn_stop]:
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            btn_row1.addWidget(btn)
        exec_layout.addLayout(btn_row1)
        
        btn_row2 = QHBoxLayout()
        self.btn_save = QPushButton("Save PNG"); self.btn_save.clicked.connect(self.save_plot)
        self.btn_csv = QPushButton("Export CSV"); self.btn_csv.clicked.connect(self.export_csv)
        for btn in [self.btn_save, self.btn_csv]:
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            btn_row2.addWidget(btn)
        exec_layout.addLayout(btn_row2)
        
        self.lbl_param_count = QLabel("Total Parameters: 0")
        self.lbl_param_count.setStyleSheet("font-weight: bold; color: #00ff00; font-size: 14px; qproperty-alignment: AlignCenter;")
        exec_layout.addWidget(self.lbl_param_count)
        
        exec_grp.setLayout(exec_layout)
        left_layout.addWidget(exec_grp)

        # 2. Configuration Tabs
        self.tabs = QTabWidget()
        left_layout.addWidget(self.tabs, stretch=1)
        
        # Helper to create a styled group box with form layout
        def create_form_group(title):
            grp = QGroupBox(title)
            form = QFormLayout()
            form.setSpacing(8)
            form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
            form.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
            grp.setLayout(form)
            return grp, form

        # --- Tab 1: Dataset ---
        tab1 = QWidget()
        tab1_layout = QVBoxLayout(tab1)
        tab1_layout.setContentsMargins(8, 8, 8, 8)
        
        grp_ds, form_ds = create_form_group("Data Distribution")
        self.combo_dataset = QComboBox(); self.combo_dataset.addItems(["Circles", "Moons", "XOR", "Spirals"])
        self.spin_seed = QSpinBox(); self.spin_seed.setRange(0, 9999); self.spin_seed.setValue(42)
        self.spin_noise = QDoubleSpinBox(); self.spin_noise.setRange(0.0, 2.0); self.spin_noise.setSingleStep(0.05); self.spin_noise.setValue(0.15)
        self.slider_imbalance = QSlider(Qt.Horizontal); self.slider_imbalance.setRange(10, 90); self.slider_imbalance.setValue(50)
        self.lbl_imbalance = QLabel("50% / 50%")
        self.slider_imbalance.valueChanged.connect(lambda v: self.lbl_imbalance.setText(f"{100-v}% / {v}% Cls 1"))
        
        form_ds.addRow("Type:", self.combo_dataset)
        form_ds.addRow("Seed:", self.spin_seed)
        form_ds.addRow("Noise:", self.spin_noise)
        form_ds.addRow("Imbalance:", self.slider_imbalance)
        form_ds.addRow("", self.lbl_imbalance)
        tab1_layout.addWidget(grp_ds)

        grp_feat, form_feat = create_form_group("Feature Engineering")
        self.combo_features = QComboBox(); self.combo_features.addItems(["Linear", "Poly2", "Poly3"])
        self.combo_features.setToolTip("Linear: (x,y)\nPoly2: (x,y,x²,y²)\nPoly3: (x,y,x²,y²,x*y)")
        self.combo_features.currentTextChanged.connect(self.update_param_count)
        form_feat.addRow("Feature Map:", self.combo_features)
        tab1_layout.addWidget(grp_feat)
        tab1_layout.addStretch(1)
        self.tabs.addTab(tab1, "Dataset")

        # --- Tab 2: Model ---
        tab2 = QWidget()
        tab2_layout = QVBoxLayout(tab2)
        tab2_layout.setContentsMargins(8, 8, 8, 8)
        
        grp_base, form_base = create_form_group("Base Architecture")
        self.combo_arch = QComboBox(); self.combo_arch.addItems(["MLP", "Transformer", "Mamba", "Autoencoder"])
        self.combo_arch.currentTextChanged.connect(self.update_param_count)
        self.combo_init = QComboBox(); self.combo_init.addItems(["Default", "Xavier", "Kaiming"])
        form_base.addRow("Model:", self.combo_arch)
        form_base.addRow("Init:", self.combo_init)
        tab2_layout.addWidget(grp_base)

        grp_custom, form_custom = create_form_group("Custom MLP (Overrides Below)")
        self.txt_custom_mlp = QLineEdit(); self.txt_custom_mlp.setPlaceholderText("e.g. 32,ReLU,16,Tanh")
        self.txt_custom_mlp.setToolTip("Define layers separated by commas.\nNumbers = layer size.\nText = activation function.")
        self.txt_custom_mlp.textChanged.connect(self.update_param_count)
        form_custom.addRow("Layers:", self.txt_custom_mlp)
        tab2_layout.addWidget(grp_custom)

        grp_fallback, form_fallback = create_form_group("Fallback MLP Config")
        self.combo_act = QComboBox(); self.combo_act.addItems(["Tanh", "ReLU", "LeakyReLU", "ELU", "Sigmoid"])
        self.spin_hidden = QSpinBox(); self.spin_hidden.setRange(4, 256); self.spin_hidden.setValue(32); self.spin_hidden.valueChanged.connect(self.update_param_count)
        self.spin_layers = QSpinBox(); self.spin_layers.setRange(1, 8); self.spin_layers.setValue(3); self.spin_layers.valueChanged.connect(self.update_param_count)
        self.spin_dropout = QDoubleSpinBox(); self.spin_dropout.setRange(0.0, 0.9); self.spin_dropout.setSingleStep(0.1); self.spin_dropout.setValue(0.0)
        
        # Container for checkboxes
        chk_container = QWidget()
        chk_layout = QHBoxLayout(chk_container)
        chk_layout.setContentsMargins(0, 0, 0, 0)
        self.chk_bn = QCheckBox("BatchNorm"); self.chk_bn.toggled.connect(self.update_param_count)
        self.chk_res = QCheckBox("Residuals"); self.chk_res.toggled.connect(self.update_param_count)
        chk_layout.addWidget(self.chk_bn)
        chk_layout.addWidget(self.chk_res)
        chk_layout.addStretch()
        
        form_fallback.addRow("Act:", self.combo_act)
        form_fallback.addRow("Size:", self.spin_hidden)
        form_fallback.addRow("Layers:", self.spin_layers)
        form_fallback.addRow("Dropout:", self.spin_dropout)
        form_fallback.addRow("Extras:", chk_container)
        tab2_layout.addWidget(grp_fallback)
        tab2_layout.addStretch(1)
        self.tabs.addTab(tab2, "Model")

        # --- Tab 3: Training ---
        tab3 = QWidget()
        tab3_layout = QVBoxLayout(tab3)
        tab3_layout.setContentsMargins(8, 8, 8, 8)
        
        grp_opt, form_opt = create_form_group("Optimizer & Loss")
        self.combo_opt = QComboBox(); self.combo_opt.addItems(["Adam", "SGD", "RMSprop"])
        self.spin_lr = QDoubleSpinBox(); self.spin_lr.setRange(0.0001, 1.0); self.spin_lr.setSingleStep(0.001); self.spin_lr.setDecimals(4); self.spin_lr.setValue(0.01)
        self.combo_loss = QComboBox(); self.combo_loss.addItems(["BCE", "Hinge (SVM)", "Focal"])
        self.spin_batch = QSpinBox(); self.spin_batch.setRange(8, 512); self.spin_batch.setSingleStep(8); self.spin_batch.setValue(32)
        form_opt.addRow("Optimizer:", self.combo_opt)
        form_opt.addRow("LR:", self.spin_lr)
        form_opt.addRow("Loss:", self.combo_loss)
        form_opt.addRow("Batch:", self.spin_batch)
        tab3_layout.addWidget(grp_opt)

        grp_reg, form_reg = create_form_group("Regularization & Stability")
        self.spin_wd = QDoubleSpinBox(); self.spin_wd.setRange(0.0, 1.0); self.spin_wd.setSingleStep(0.001); self.spin_wd.setDecimals(4); self.spin_wd.setValue(0.0)
        self.spin_aug = QDoubleSpinBox(); self.spin_aug.setRange(0.0, 1.0); self.spin_aug.setSingleStep(0.05); self.spin_aug.setValue(0.0)
        self.spin_clip = QDoubleSpinBox(); self.spin_clip.setRange(0.0, 10.0); self.spin_clip.setSingleStep(0.5); self.spin_clip.setValue(0.0)
        self.spin_aug.setToolTip("Injects Gaussian noise into training batches on the fly.")
        self.spin_clip.setToolTip("Clips gradient L2 norm to prevent exploding gradients.")
        form_reg.addRow("Weight Decay:", self.spin_wd)
        form_reg.addRow("Aug Noise:", self.spin_aug)
        form_reg.addRow("Grad Clip:", self.spin_clip)
        tab3_layout.addWidget(grp_reg)

        grp_sched, form_sched = create_form_group("Schedulers & UI")
        self.spin_es = QSpinBox(); self.spin_es.setRange(0, 500); self.spin_es.setValue(0); self.spin_es.setSpecialValueText("Off")
        self.spin_update = QSpinBox(); self.spin_update.setRange(1, 200); self.spin_update.setValue(10)
        self.chk_scheduler = QCheckBox("Cosine Annealing LR")
        form_sched.addRow("Early Stop:", self.spin_es)
        form_sched.addRow("UI Update:", self.spin_update)
        form_sched.addRow("Scheduler:", self.chk_scheduler)
        tab3_layout.addWidget(grp_sched)
        tab3_layout.addStretch(1)
        self.tabs.addTab(tab3, "Training")

        main_layout.addWidget(left_panel)

        # --- Right Matplotlib Canvas ---
        self.fig = Figure(figsize=(10, 8), constrained_layout=True)
        self.fig.patch.set_facecolor('#1e1e1e')
        self.canvas = FigureCanvas(self.fig)
        main_layout.addWidget(self.canvas, 4)

        self.ax_heatmap = self.fig.add_subplot(3, 2, 1)
        self.ax_heightmap = self.fig.add_subplot(3, 2, 2, projection='3d')
        self.ax_w1 = self.fig.add_subplot(3,2, 3)
        self.ax_info = self.fig.add_subplot(3, 2, 4) 
        self.ax_loss = self.fig.add_subplot(3, 1, 3)
        self.ax_acc = self.ax_loss.twinx() 
        
        self.surf = None
        self.wire = None
        self.cbar_heatmap = None
        self.worker = None
        self._is_paused_state = False
        self.history_df = None
        self.grad_history = [] 
        
        self._init_plots()
        self.update_param_count()
        self.statusBar().showMessage("Ready. Configure settings and press Start.")

    def _init_plots(self):
        for ax in [self.ax_heatmap, self.ax_w1, self.ax_info, self.ax_loss]:
            ax.set_facecolor('#2b2b2b')
            ax.title.set_color('white'); ax.xaxis.label.set_color('white'); ax.yaxis.label.set_color('white')
            ax.tick_params(colors='white')
            for spine in ax.spines.values(): spine.set_color('#555555')
                
        self.ax_heatmap.set_title("Probability Topography & Decision Boundary")
        self.ax_heightmap.set_title("3D Logits Surface & Data Geometry")
        self.ax_w1.set_title("Layer 1 Weight Distribution")
        self.ax_info.set_title("Gradient Flow / Latent Space")
        self.ax_loss.set_title("Training vs Validation Metrics")
        self.ax_loss.set_xlabel("Epochs"); self.ax_loss.set_ylabel("Loss (Log)", color='red')
        self.ax_acc.set_ylabel("Accuracy", color='cyan')
        
        self.ax_loss.grid(True, which="both", linestyle='--', alpha=0.3, color='gray')
        self.canvas.draw_idle()

    def update_param_count(self):
        mode = self.combo_features.currentText()
        input_dim = 2 if mode == "Linear" else (4 if mode == "Poly2" else 5)
        arch = self.combo_arch.currentText()
        if arch == "MLP": model = MLPPreset(input_dim, self.txt_custom_mlp.text(), self.spin_hidden.value(), self.spin_layers.value(), "Tanh", 0.0, self.chk_bn.isChecked(), self.chk_res.isChecked())
        elif arch == "Transformer": model = TransformerPreset()
        elif arch == "Mamba": model = MambaPreset()
        else: model = AutoencoderPreset()
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.lbl_param_count.setText(f"Total Parameters: {params:,}")

    def start_training(self):
        if self.worker and self.worker.isRunning(): return
        self.btn_start.setEnabled(False); self.btn_pause.setEnabled(True); self.btn_stop.setEnabled(True)
        self.btn_pause.setText("Pause"); self._is_paused_state = False
        self.tabs.setEnabled(False)
        
        mode = self.combo_features.currentText()
        input_dim = 2 if mode == "Linear" else (4 if mode == "Poly2" else 5)
        config = {
            'dataset_type': self.combo_dataset.currentText(), 'architecture': self.combo_arch.currentText(),
            'lr': self.spin_lr.value(), 'batch_size': self.spin_batch.value(), 'seed': self.spin_seed.value(),
            'noise': self.spin_noise.value(), 'imbalance_ratio': self.slider_imbalance.value() / 100.0,
            'feature_mode': mode, 'input_dim': input_dim, 'custom_mlp': self.txt_custom_mlp.text(),
            'mlp_activation': self.combo_act.currentText(), 'mlp_hidden': self.spin_hidden.value(),
            'mlp_layers': self.spin_layers.value(), 'dropout': self.spin_dropout.value(),
            'use_bn': self.chk_bn.isChecked(), 'use_res': self.chk_res.isChecked(),
            'init_method': self.combo_init.currentText(), 'optimizer_name': self.combo_opt.currentText(),
            'weight_decay': self.spin_wd.value(), 'aug_noise': self.spin_aug.value(),
            'grad_clip': self.spin_clip.value(), 'loss_func': self.combo_loss.currentText(),
            'use_scheduler': self.chk_scheduler.isChecked(), 'early_stopping': self.spin_es.value(),
            'update_interval': self.spin_update.value()
        }
        self.worker = TrainingWorker(config)
        self.worker.update_plots.connect(self.update_visualization)
        self.worker.finished_training.connect(self.training_finished)
        self.worker.start()
        self.statusBar().showMessage("Training started...")

    def toggle_pause(self):
        if not self.worker: return
        self._is_paused_state = not self._is_paused_state
        self.worker.set_paused(self._is_paused_state)
        self.btn_pause.setText("Resume" if self._is_paused_state else "Pause")
        self.statusBar().showMessage("Training Paused." if self._is_paused_state else "Training Resumed.")

    def stop_training(self):
        if self.worker and self.worker.isRunning(): self.worker.stop(); self.worker.wait(2000)

    def save_plot(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Plot", "nn_snapshot.png", "PNG Images (*.png)")
        if file_path: self.fig.savefig(file_path, dpi=300, facecolor=self.fig.get_facecolor()); self.statusBar().showMessage(f"Plot saved to {file_path}")

    def export_csv(self):
        if self.history_df is None: self.statusBar().showMessage("No training history available yet."); return
        file_path, _ = QFileDialog.getSaveFileName(self, "Export CSV", "training_history.csv", "CSV Files (*.csv)")
        if file_path: self.history_df.to_csv(file_path, index=False); self.statusBar().showMessage(f"History exported to {file_path}")

    def update_visualization(self, p):
        arch_name = self.worker.config['architecture']
        loss_func = self.worker.config['loss_func']
        opt_name = self.worker.config['optimizer_name']
        wd = self.worker.config['weight_decay']
        
        y_np = self.worker.y.numpy().flatten()
        X_data = self.worker.X_orig[:, 0]; Y_data = self.worker.X_orig[:, 1]
        X_grid = self.worker.X_grid; Y_grid = self.worker.Y_grid
        train_mask = self.worker.train_mask.numpy()
        misclassified = p['misclassified_mask']
        data_logits = p['data_logits']

        # --- 1. Probability Heatmap ---
        self.ax_heatmap.clear()
        self.ax_heatmap.set_facecolor('#2b2b2b')
        
        if loss_func == "Hinge": P = (p['Z'] / 5.0 + 1.0) / 2.0 
        else: P = 1 / (1 + np.exp(-p['Z']))
            
        im = self.ax_heatmap.imshow(P, extent=[X_grid.min(), X_grid.max(), Y_grid.min(), Y_grid.max()], 
                                    origin='lower', cmap='RdBu_r', alpha=0.85, vmin=0, vmax=1, interpolation='bicubic')
        
        cs = self.ax_heatmap.contour(X_grid, Y_grid, P, levels=[0.1, 0.3, 0.7, 0.9], colors='gray', linewidths=0.8, alpha=0.6)
        self.ax_heatmap.clabel(cs, inline=True, fontsize=8, fmt='%.1f', colors='white')
        self.ax_heatmap.contour(X_grid, Y_grid, P, levels=[0.5], colors='lime', linewidths=3, linestyles='solid')
        
        self.ax_heatmap.scatter(X_data[train_mask & (y_np==0)], Y_data[train_mask & (y_np==0)], facecolors='none', edgecolors='blue', s=40, linewidths=1.5)
        self.ax_heatmap.scatter(X_data[train_mask & (y_np==1)], Y_data[train_mask & (y_np==1)], facecolors='none', edgecolors='red', s=40, linewidths=1.5)
        self.ax_heatmap.scatter(X_data[~train_mask & (y_np==0)], Y_data[~train_mask & (y_np==0)], c='blue', marker='s', edgecolors='white', s=40)
        self.ax_heatmap.scatter(X_data[~train_mask & (y_np==1)], Y_data[~train_mask & (y_np==1)], c='red', marker='s', edgecolors='white', s=40)
        
        if misclassified.any():
            self.ax_heatmap.scatter(X_data[misclassified], Y_data[misclassified], marker='X', c='yellow', s=120, linewidths=1.5, edgecolors='black', label='Misclassified', zorder=5)
            self.ax_heatmap.legend(loc='upper right', fontsize=8)
            
        txt = f"Opt: {opt_name} | WD: {wd} | Loss: {loss_func}\nTr:{p['train_acc']:.2f} | Val:{p['val_acc']:.2f} (E:{p['epoch']})"
        self.ax_heatmap.text(0.02, 0.98, txt, transform=self.ax_heatmap.transAxes, fontsize=8, va='top', color='white', bbox=dict(facecolor='#1e1e1e', alpha=0.7, edgecolor='none'))
        
        if self.cbar_heatmap: self.cbar_heatmap.remove()
        self.cbar_heatmap = self.fig.colorbar(im, ax=self.ax_heatmap, shrink=0.8, aspect=15)
        self.cbar_heatmap.set_label('Probability (Class 1)', color='white')
        self.cbar_heatmap.ax.tick_params(colors='white')
        self.cbar_heatmap.outline.set_edgecolor('white')

        # --- 2. 3D Heightmap ---
        if self.surf: self.surf.remove()
        if self.wire: self.wire.remove()
        self.ax_heightmap.clear()
        
        self.ax_heightmap.set_xlim(-5, 5); self.ax_heightmap.set_ylim(-5, 5); self.ax_heightmap.set_zlim(-5, 5)
        self.ax_heightmap.view_init(elev=30, azim=-60)
        
        self.ax_heightmap.xaxis.pane.fill = False
        self.ax_heightmap.yaxis.pane.fill = False
        self.ax_heightmap.zaxis.pane.fill = False
        self.ax_heightmap.xaxis.pane.set_edgecolor('gray')
        self.ax_heightmap.yaxis.pane.set_edgecolor('gray')
        self.ax_heightmap.zaxis.pane.set_edgecolor('gray')
        self.ax_heightmap.grid(True, alpha=0.3, color='gray')
        
        norm = Normalize(vmin=-5, vmax=5)
        self.surf = self.ax_heightmap.plot_surface(X_grid, Y_grid, p['Z'], cmap='RdBu_r', edgecolor='none', alpha=0.85, norm=norm)
        self.wire = self.ax_heightmap.plot_wireframe(X_grid, Y_grid, p['Z'], color='black', alpha=0.15, linewidth=0.5)
        self.ax_heightmap.contour(X_grid, Y_grid, p['Z'], levels=[0.0], zdir='z', offset=0, colors='lime', linewidths=3, alpha=0.9)
        
        z_data = np.clip(data_logits, -5, 5)
        self.ax_heightmap.scatter(X_data[train_mask], Y_data[train_mask], z_data[train_mask], c=y_np[train_mask], cmap='bwr', marker='o', edgecolors='black', s=20, depthshade=False)
        self.ax_heightmap.scatter(X_data[~train_mask], Y_data[~train_mask], z_data[~train_mask], c=y_np[~train_mask], cmap='bwr', marker='s', edgecolors='white', s=25, depthshade=False)
        
        self.ax_heightmap.set_title("3D Logits Surface & Data Geometry", color='white')
        self.ax_heightmap.tick_params(colors='white')

        # --- 3. Weights Histogram ---
        self.ax_w1.clear()
        self.ax_w1.set_facecolor('#2b2b2b')
        if len(p['w1']) > 0:
            self.ax_w1.hist(p['w1'], bins=50, color='cyan', alpha=0.7)
            self.ax_w1.axvline(p['w1'].mean(), color='white', linestyle='dashed', linewidth=1.5)
        self.ax_w1.set_title(f"Layer 1 Weights (Init: {self.combo_init.currentText()})", fontsize=9)
        self.ax_w1.tick_params(colors='white')
        for spine in self.ax_w1.spines.values(): spine.set_color('#555555')

        # --- 4. Dynamic Info Plot ---
        self.ax_info.clear()
        self.ax_info.set_facecolor('#2b2b2b')
        
        if arch_name == "Autoencoder" and p['latent'] is not None:
            latent = p['latent']; y_val = p['latent_y']
            x_min, x_max = latent[:, 0].min()-1, latent[:, 0].max()+1
            y_min, y_max = latent[:, 1].min()-1, latent[:, 1].max()+1
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50), np.linspace(y_min, y_max, 50))
            grid_latent = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])
            with torch.no_grad():
                latent_logits = self.worker.model.classifier(grid_latent)
                latent_Z = latent_logits.numpy().reshape(xx.shape)
            
            self.ax_info.contourf(xx, yy, latent_Z, levels=[-10, 0, 10], cmap='bwr', alpha=0.3)
            self.ax_info.contour(xx, yy, latent_Z, levels=[0.0], colors='lime', linewidths=2)
            self.ax_info.scatter(latent[:, 0], latent[:, 1], c=y_val, cmap='bwr', edgecolors='k', s=30)
            self.ax_info.set_title("Latent Space & Classifier Boundary", fontsize=9)
            self.ax_info.set_xlabel("Z1"); self.ax_info.set_ylabel("Z2")
        else:
            self.grad_history.append(p['grads'])
            if len(self.grad_history) > 50: self.grad_history.pop(0)
            for i in range(len(p['grads'])):
                vals = [h[i] for h in self.grad_history]
                self.ax_info.plot(range(len(vals)), vals, marker='o', linestyle='-', alpha=0.8, label=f'L{i}')
            self.ax_info.set_title("Gradient Flow History (L2 Norm)", fontsize=9)
            self.ax_info.set_xlabel("Update Steps (Rolling 50)"); self.ax_info.set_ylabel("L2 Norm (Log)")
            self.ax_info.set_yscale('log')
            if len(p['grads']) <= 6: self.ax_info.legend(loc='upper right', fontsize=7)
            
        self.ax_info.tick_params(colors='white')
        for spine in self.ax_info.spines.values(): spine.set_color('#555555')

        # --- 5. Loss & Accuracy Curves ---
        self.ax_loss.clear()
        self.ax_loss.set_facecolor('#2b2b2b')
        self.ax_loss.plot(p['loss_hist'], 'r-', label='Train Loss', alpha=0.8, linewidth=2)
        self.ax_loss.plot(p['val_loss_hist'], 'm--', label='Val Loss', alpha=0.8, linewidth=2)
        self.ax_loss.set_ylabel("Loss (Log)", color='red'); self.ax_loss.set_yscale('log')
        self.ax_loss.tick_params(axis='y', labelcolor='red')
        self.ax_loss.grid(True, which="both", linestyle='--', alpha=0.3, color='gray')
        
        self.ax_acc.clear()
        self.ax_acc.plot(p['acc_hist'], 'c-', label='Train Acc', alpha=0.8, linewidth=2)
        self.ax_acc.plot(p['val_acc_hist'], 'b--', label='Val Acc', alpha=0.8, linewidth=2)
        self.ax_acc.set_ylabel("Accuracy", color='cyan'); self.ax_acc.set_ylim(0, 1.05)
        self.ax_acc.tick_params(axis='y', labelcolor='cyan')
        
        self.ax_loss.set_title(f"Metrics (LR: {p['lr']:.4f})")
        self.ax_loss.set_xlabel("Epochs")
        self.ax_loss.legend(loc='upper left', fontsize=8)
        self.ax_acc.legend(loc='lower right', fontsize=8)
        
        self.ax_loss.tick_params(colors='white')
        self.ax_loss.xaxis.label.set_color('white'); self.ax_loss.yaxis.label.set_color('white'); self.ax_loss.title.set_color('white')
        for spine in self.ax_loss.spines.values(): spine.set_color('#555555')
        for spine in self.ax_acc.spines.values(): spine.set_color('#555555')

        self.canvas.draw_idle()
        
        self.history_df = pd.DataFrame({
            'Epoch': range(len(p['loss_hist'])), 'Train Loss': p['loss_hist'], 'Val Loss': p['val_loss_hist'],
            'Train Acc': p['acc_hist'], 'Val Acc': p['val_acc_hist']
        })
        if not self._is_paused_state:
            self.statusBar().showMessage(f"Ep: {p['epoch']} | TrL: {p['train_loss']:.4f} | VaL: {p['val_loss']:.4f} | TrA: {p['train_acc']:.4f} | VaA: {p['val_acc']:.4f} | LR: {p['lr']:.5f}")

    def training_finished(self, stop_reason):
        self.btn_start.setEnabled(True); self.btn_start.setText("Restart")
        self.btn_pause.setEnabled(False); self.btn_stop.setEnabled(False); self.tabs.setEnabled(True)
        self.statusBar().showMessage(f"Training Halted. Reason: {stop_reason}")

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning(): self.worker.stop(); self.worker.wait(2000)
        event.accept()

def apply_dark_theme(app):
    dark_palette = QPalette()
    dark_palette.setColor(QPalette.Window, QColor(30, 30, 30)); dark_palette.setColor(QPalette.WindowText, Qt.white)
    dark_palette.setColor(QPalette.Base, QColor(45, 45, 45)); dark_palette.setColor(QPalette.AlternateBase, QColor(30, 30, 30))
    dark_palette.setColor(QPalette.ToolTipBase, Qt.white); dark_palette.setColor(QPalette.ToolTipText, Qt.white)
    dark_palette.setColor(QPalette.Text, Qt.white); dark_palette.setColor(QPalette.Button, QColor(45, 45, 45))
    dark_palette.setColor(QPalette.ButtonText, Qt.white); dark_palette.setColor(QPalette.BrightText, Qt.red)
    dark_palette.setColor(QPalette.Link, QColor(42, 130, 218)); dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(dark_palette); app.setStyle("Fusion")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    apply_dark_theme(app)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
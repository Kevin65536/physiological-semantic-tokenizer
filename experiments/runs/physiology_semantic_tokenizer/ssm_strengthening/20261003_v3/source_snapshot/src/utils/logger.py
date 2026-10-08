"""
Experiment logger for physiology-semantic runs.
Handles config management, metrics logging, and visualization generation.
"""

import json
import hashlib
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Iterator, Optional, List
import numpy as np
import yaml

from .checkpoints import save_checkpoint_payload
from .io import write_json, write_yaml
from src.data.registry import load_experiment_config

# Try to import visualization libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def _json_safe_metric_value(value: Any) -> Any:
    """Convert common scalar-like metric values into JSON-safe values."""
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, 'detach'):
        value = value.detach()
    if hasattr(value, 'cpu'):
        value = value.cpu()
    if hasattr(value, 'item'):
        try:
            return value.item()
        except (TypeError, ValueError):
            return str(value)
    return value


def _sanitize_metrics(metrics: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return {str(key): _json_safe_metric_value(value) for key, value in (metrics or {}).items()}


def _format_metric_for_log(key: str, value: Any) -> str:
    del key
    if isinstance(value, (int, float, np.number)) and not isinstance(value, bool):
        return f"{float(value):.4f}"
    return str(value)


class ExperimentLogger:
    """Logger for tracking experiment configurations, metrics, and results."""
    
    def __init__(
        self,
        config_path: str,
        experiments_dir: str = "experiments",
        run_name: Optional[str] = None
    ):
        """
        Initialize experiment logger.
        
        Args:
            config_path: Path to experiment config YAML file
            experiments_dir: Base directory for experiments
            run_name: Optional custom run name (default: auto-generated)
        """
        self.experiments_dir = Path(experiments_dir)
        self.configs_dir = self.experiments_dir / "configs"
        self.runs_dir = self.experiments_dir / "runs"
        self.results_dir = self.experiments_dir / "results"
        
        # Load and merge config
        self.config = self._load_config(config_path)
        self.config_hash = self._hash_config(self.config)
        
        # Create run directory
        exp_name = self.config.get("experiment", {}).get("name", "unknown")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = run_name or f"{timestamp}_{exp_name}"
        self.run_root = self._resolve_run_root()
        self.run_dir = self.run_root / self.run_name
        metrics_path = self.run_dir / "metrics.json"
        existing_run = self.run_dir.exists()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.checkpoints_dir = self.run_dir / "checkpoints"
        self.figures_dir = self.run_dir / "figures"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metrics storage
        if existing_run and metrics_path.exists():
            with open(metrics_path, 'r', encoding='utf-8') as f:
                self.metrics = json.load(f)
            self.metrics["config_hash"] = self.config_hash
            self.metrics.setdefault("epochs", [])
            self.metrics.setdefault("final_metrics", {})
            resumed_at = self.metrics.setdefault("resumed_at", [])
            resumed_at.append(datetime.now().isoformat())
            self.metrics["completed_at"] = None
        else:
            self.metrics = {
                "config_hash": self.config_hash,
                "started_at": datetime.now().isoformat(),
                "completed_at": None,
                "epochs": [],
                "final_metrics": {}
            }
        
        # Save frozen config
        self._save_config()
        
        if existing_run and metrics_path.exists():
            print(f"[ExperimentLogger] Reusing run: {self.run_name}")
        else:
            print(f"[ExperimentLogger] Initialized run: {self.run_name}")
        print(f"[ExperimentLogger] Run directory: {self.run_dir}")

    def _resolve_run_root(self) -> Path:
        """Resolve the namespace used for this experiment's run directory."""
        run_group = self.config.get("experiment", {}).get("run_group")
        if not run_group:
            return self.runs_dir

        group_path = Path(str(run_group).replace("\\", "/"))
        if group_path.is_absolute() or ".." in group_path.parts:
            raise ValueError(
                "experiment.run_group must be a relative path below experiments/runs"
            )
        return self.runs_dir / group_path
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load config through the shared experiment config interface."""
        return load_experiment_config(config_path, configs_dir=self.configs_dir)
    
    def _merge_configs(self, base: Dict, override: Dict) -> Dict:
        """Deep merge two config dictionaries."""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        return result
    
    def _hash_config(self, config: Dict) -> str:
        """Generate hash of config for reproducibility tracking."""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    def _save_config(self):
        """Save frozen config to run directory."""
        write_yaml(self.run_dir / "config.yaml", self.config)
    
    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float] = None,
        loss_breakdown: Optional[Dict[str, float]] = None,
        metrics: Optional[Dict[str, float]] = None
    ):
        """
        Log metrics for an epoch.
        
        Args:
            epoch: Current epoch number
            train_loss: Total training loss
            val_loss: Validation loss (optional)
            loss_breakdown: Dict of individual loss components
            metrics: Dict of evaluation metrics (correlations, HSIC, etc.)
        """
        safe_loss_breakdown = _sanitize_metrics(loss_breakdown)
        safe_metrics = _sanitize_metrics(metrics)
        epoch_data = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss) if val_loss is not None else None,
            "loss_breakdown": safe_loss_breakdown,
            "metrics": safe_metrics,
            "scalars": dict(safe_metrics),
        }
        for key, value in safe_metrics.items():
            if key not in epoch_data:
                epoch_data[key] = value
        self.metrics["epochs"].append(epoch_data)
        self._save_metrics()
        
        # Print progress
        metrics_str = ", ".join(f"{k}: {_format_metric_for_log(k, v)}" for k, v in safe_metrics.items())
        if val_loss is not None:
            print(f"[Epoch {epoch}] train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f} | {metrics_str}")
        else:
            print(f"[Epoch {epoch}] train_loss: {train_loss:.4f} | {metrics_str}")
    
    def log_final(self, final_metrics: Dict[str, float]):
        """Log final evaluation metrics."""
        # 处理混合类型的metrics (数值和字符串)
        processed_metrics = {}
        for k, v in final_metrics.items():
            if isinstance(v, str):
                processed_metrics[k] = v
            elif isinstance(v, (int, float)):
                processed_metrics[k] = float(v)
            elif hasattr(v, 'item'):  # tensor
                processed_metrics[k] = float(v.item())
            else:
                processed_metrics[k] = v
        self.metrics["final_metrics"] = processed_metrics
        self.metrics["completed_at"] = datetime.now().isoformat()
        self._save_metrics()
    
    def _save_metrics(self):
        """Save metrics to JSON file."""
        write_json(self.run_dir / "metrics.json", self.metrics)
    
    def save_checkpoint(
        self,
        state_dict: Dict,
        epoch: int,
        is_best: bool = False,
        keep_epoch_copy: bool = True,
    ):
        """Save model checkpoint."""
        # Ensure directory exists
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = self.checkpoints_dir / f"checkpoint_epoch_{epoch}.pt"
        if keep_epoch_copy:
            save_checkpoint_payload(state_dict, checkpoint_path)

        if is_best:
            best_path = self.checkpoints_dir / "best_model.pt"
            if keep_epoch_copy:
                shutil.copy(checkpoint_path, best_path)
            else:
                save_checkpoint_payload(state_dict, best_path)
    
    def generate_figures(self):
        """Generate visualization figures from logged metrics."""
        if not HAS_MATPLOTLIB:
            print("[Warning] matplotlib not available, skipping figure generation")
            return

        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.metrics["epochs"]:
            print("[Warning] No epochs logged, skipping figure generation")
            return
        
        epochs = [e["epoch"] for e in self.metrics["epochs"]]
        
        # 1. Training curves
        self._plot_training_curves(epochs)
        
        # 2. Latent recovery (if metrics available)
        if self.metrics["epochs"][0].get("metrics"):
            self._plot_latent_recovery(epochs)
        
        # 3. Loss breakdown
        if self.metrics["epochs"][0].get("loss_breakdown"):
            self._plot_loss_breakdown(epochs)
        
        print(f"[ExperimentLogger] Figures saved to {self.figures_dir}")
    
    def _plot_training_curves(self, epochs: List[int]):
        """Plot training and validation loss curves."""
        train_losses = [e["train_loss"] for e in self.metrics["epochs"]]
        val_losses = [e["val_loss"] for e in self.metrics["epochs"] if e["val_loss"] is not None]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epochs, train_losses, label='Train Loss', linewidth=2)
        if val_losses and len(val_losses) == len(epochs):
            ax.plot(epochs, val_losses, label='Val Loss', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(f'Training Curves - {self.config["experiment"]["name"]}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        fig.savefig(self.figures_dir / "training_curves.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    def _plot_latent_recovery(self, epochs: List[int]):
        """Plot latent recovery correlations or other tracked metrics over training."""
        # First try the original PID metrics
        pid_metric_names = ["corr_zr_wr", "corr_zu_wu", "corr_zs_ws"]
        pid_metric_labels = ["Redundancy (z_r)", "Unique (z_u)", "Synergy (z_s)"]
        
        # Check if any PID metrics exist
        has_pid_metrics = any(
            any(e["metrics"].get(name) is not None for e in self.metrics["epochs"])
            for name in pid_metric_names
        )
        
        if has_pid_metrics:
            # Plot PID latent recovery
            fig, ax = plt.subplots(figsize=(10, 6))
            
            for name, label in zip(pid_metric_names, pid_metric_labels):
                values = [e["metrics"].get(name) for e in self.metrics["epochs"]]
                if any(v is not None for v in values):
                    valid_epochs = [ep for ep, v in zip(epochs, values) if v is not None]
                    valid_values = [v for v in values if v is not None]
                    ax.plot(valid_epochs, valid_values, label=label, linewidth=2, marker='o', markersize=3)
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Correlation with Ground Truth')
            ax.set_title('Latent Recovery')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)
            ax.axhline(y=0.6, color='r', linestyle='--', alpha=0.5, label='Target')
            
            fig.savefig(self.figures_dir / "latent_recovery.png", dpi=150, bbox_inches='tight')
            plt.close(fig)
        else:
            # Plot all available metrics dynamically
            all_metric_names = set()
            for e in self.metrics["epochs"]:
                if e.get("metrics"):
                    all_metric_names.update(e["metrics"].keys())
            
            # Filter out lr and non-numeric metrics
            numeric_metrics = []
            for name in all_metric_names:
                sample_val = next((e["metrics"].get(name) for e in self.metrics["epochs"] 
                                  if e["metrics"].get(name) is not None), None)
                if isinstance(sample_val, (int, float)):
                    numeric_metrics.append(name)
            
            if not numeric_metrics:
                return
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            for name in sorted(numeric_metrics):
                if name == 'lr':  # Skip learning rate, it's plotted separately
                    continue
                values = [e["metrics"].get(name) for e in self.metrics["epochs"]]
                if any(v is not None for v in values):
                    valid_epochs = [ep for ep, v in zip(epochs, values) if v is not None]
                    valid_values = [v for v in values if v is not None]
                    ax.plot(valid_epochs, valid_values, label=name, linewidth=2, marker='o', markersize=3)
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Value')
            ax.set_title('Training Metrics')
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)
            
            fig.savefig(self.figures_dir / "training_metrics.png", dpi=150, bbox_inches='tight')
            plt.close(fig)
    
    def _plot_loss_breakdown(self, epochs: List[int]):
        """Plot individual loss components dynamically."""
        # Collect all loss component names from the logged data
        all_loss_names = set()
        for e in self.metrics["epochs"]:
            if e.get("loss_breakdown"):
                all_loss_names.update(e["loss_breakdown"].keys())
        
        if not all_loss_names:
            return
        
        # Filter to get only actual loss components (exclude utilization, etc.)
        loss_names = [name for name in all_loss_names 
                     if 'loss' in name.lower() or name in ['reconstruction', 'alignment', 'orthogonality', 'synergy']]
        
        # If no specific loss names found, use all available
        if not loss_names:
            loss_names = list(all_loss_names)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(loss_names)))
        
        for i, name in enumerate(sorted(loss_names)):
            values = [e["loss_breakdown"].get(name) for e in self.metrics["epochs"]]
            if any(v is not None for v in values):
                valid_epochs = [ep for ep, v in zip(epochs, values) if v is not None]
                valid_values = [v for v in values if v is not None]
                # Clean up label name
                label = name.replace('_', ' ').replace('loss', '').strip().title()
                if not label:
                    label = name
                ax.plot(valid_epochs, valid_values, label=label, linewidth=2, color=colors[i])
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss Value')
        ax.set_title('Loss Components')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        fig.savefig(self.figures_dir / "loss_breakdown.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # Also create a stacked area plot if we have multiple components
        if len(loss_names) > 1:
            self._plot_loss_stacked(epochs, loss_names)
        
        # Plot utilization if available
        self._plot_utilization(epochs)
    
    def _plot_loss_stacked(self, epochs: List[int], loss_names: List[str]):
        """Plot stacked area chart of loss components."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Collect values for each loss component
        loss_data = {}
        for name in sorted(loss_names):
            values = []
            for e in self.metrics["epochs"]:
                val = e.get("loss_breakdown", {}).get(name, 0.0)
                values.append(val if val is not None else 0.0)
            loss_data[name] = values
        
        # Stack plot
        labels = [name.replace('_', ' ').replace('loss', '').strip().title() or name 
                  for name in sorted(loss_names)]
        values_list = [loss_data[name] for name in sorted(loss_names)]
        
        ax.stackplot(epochs, *values_list, labels=labels, alpha=0.7)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss Value')
        ax.set_title('Loss Components (Stacked)')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        
        fig.savefig(self.figures_dir / "loss_stacked.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    def _plot_utilization(self, epochs: List[int]):
        """Plot codebook utilization metrics over training."""
        # Check for utilization in loss_breakdown
        util_values = []
        val_util_values = []
        
        for e in self.metrics["epochs"]:
            # Try different possible keys
            util = e.get("loss_breakdown", {}).get("utilization")
            val_util = e.get("metrics", {}).get("val_utilization")
            util_values.append(util)
            val_util_values.append(val_util)
        
        has_util = any(v is not None for v in util_values)
        has_val_util = any(v is not None for v in val_util_values)
        
        if not has_util and not has_val_util:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if has_util:
            valid_epochs = [ep for ep, v in zip(epochs, util_values) if v is not None]
            valid_values = [v for v in util_values if v is not None]
            ax.plot(valid_epochs, [v * 100 for v in valid_values], 
                   label='Train Utilization', linewidth=2, color='#2E86AB')
        
        if has_val_util:
            valid_epochs = [ep for ep, v in zip(epochs, val_util_values) if v is not None]
            valid_values = [v for v in val_util_values if v is not None]
            ax.plot(valid_epochs, [v * 100 for v in valid_values], 
                   label='Val Utilization', linewidth=2, color='#A23B72', linestyle='--')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Utilization (%)')
        ax.set_title('Codebook Utilization')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 105])
        ax.axhline(y=90, color='green', linestyle=':', alpha=0.5, label='Target: 90%')
        
        fig.savefig(self.figures_dir / "utilization.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    def plot_signal_reconstruction(
        self, 
        x1_orig: np.ndarray, 
        x2_orig: np.ndarray,
        x1_rec: np.ndarray,
        x2_rec: np.ndarray,
        sample_idx: int = 0,
        fs: float = 200.0
    ):
        """
        Plot original vs reconstructed signals.
        
        Args:
            x1_orig: Original X1 signal [batch, seq] or [seq]
            x2_orig: Original X2 signal
            x1_rec: Reconstructed X1 signal
            x2_rec: Reconstructed X2 signal
            sample_idx: Which sample to plot (if batch)
            fs: Sampling frequency for time axis
        """
        if not HAS_MATPLOTLIB:
            return
        
        # Handle batch dimension
        if x1_orig.ndim == 2:
            x1_orig = x1_orig[sample_idx]
            x2_orig = x2_orig[sample_idx]
            x1_rec = x1_rec[sample_idx]
            x2_rec = x2_rec[sample_idx]
        
        seq_len = len(x1_orig)
        t = np.arange(seq_len) / fs
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        
        # X1 time domain
        axes[0, 0].plot(t, x1_orig, label='Original', alpha=0.8, linewidth=1)
        axes[0, 0].plot(t, x1_rec, label='Reconstructed', alpha=0.8, linewidth=1)
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Amplitude')
        axes[0, 0].set_title('X1 (EEG-like) - Time Domain')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # X2 time domain
        axes[0, 1].plot(t, x2_orig, label='Original', alpha=0.8, linewidth=1)
        axes[0, 1].plot(t, x2_rec, label='Reconstructed', alpha=0.8, linewidth=1)
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Amplitude')
        axes[0, 1].set_title('X2 (fNIRS-like) - Time Domain')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # X1 frequency domain
        freqs = np.fft.fftfreq(seq_len, 1/fs)[:seq_len//2]
        x1_fft_orig = np.abs(np.fft.fft(x1_orig))[:seq_len//2]
        x1_fft_rec = np.abs(np.fft.fft(x1_rec))[:seq_len//2]
        axes[1, 0].plot(freqs, x1_fft_orig, label='Original', alpha=0.8)
        axes[1, 0].plot(freqs, x1_fft_rec, label='Reconstructed', alpha=0.8)
        axes[1, 0].set_xlabel('Frequency (Hz)')
        axes[1, 0].set_ylabel('Magnitude')
        axes[1, 0].set_title('X1 - Frequency Domain')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_xlim(0, fs/4)  # Show up to Nyquist/2
        
        # X2 frequency domain
        x2_fft_orig = np.abs(np.fft.fft(x2_orig))[:seq_len//2]
        x2_fft_rec = np.abs(np.fft.fft(x2_rec))[:seq_len//2]
        axes[1, 1].plot(freqs, x2_fft_orig, label='Original', alpha=0.8)
        axes[1, 1].plot(freqs, x2_fft_rec, label='Reconstructed', alpha=0.8)
        axes[1, 1].set_xlabel('Frequency (Hz)')
        axes[1, 1].set_ylabel('Magnitude')
        axes[1, 1].set_title('X2 - Frequency Domain')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_xlim(0, fs/4)
        
        plt.tight_layout()
        fig.savefig(self.figures_dir / "signal_reconstruction.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    def plot_pid_analysis(
        self,
        z_r: np.ndarray,
        z_u1: np.ndarray,
        z_u2: np.ndarray,
        z_s: np.ndarray,
        w_r: np.ndarray,
        w_u1: np.ndarray,
        w_u2: np.ndarray,
        w_s: np.ndarray,
        fs: float = 200.0
    ):
        """
        Plot detailed PID component analysis.
        
        Shows:
        - Token distributions (histograms)
        - Ground truth vs learned correlations
        - Frequency content of tokens
        - Cross-correlation matrix
        """
        if not HAS_MATPLOTLIB:
            return
        
        # Flatten for analysis if needed
        def flatten(x):
            if x.ndim > 1:
                return x.mean(axis=-1) if x.shape[-1] > 1 else x.flatten()
            return x
        
        z_r_flat = flatten(z_r)
        z_u1_flat = flatten(z_u1)
        z_s_flat = flatten(z_s)
        w_r_flat = flatten(w_r)
        w_u1_flat = flatten(w_u1)
        w_s_flat = flatten(w_s)
        
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Token value distributions (top row)
        ax1 = fig.add_subplot(3, 3, 1)
        ax1.hist(z_r_flat, bins=50, alpha=0.7, label='z_r (learned)', density=True)
        ax1.hist(w_r_flat, bins=50, alpha=0.7, label='w_r (ground truth)', density=True)
        ax1.set_title('Redundancy Distribution')
        ax1.legend()
        ax1.set_xlabel('Value')
        
        ax2 = fig.add_subplot(3, 3, 2)
        ax2.hist(z_u1_flat, bins=50, alpha=0.7, label='z_u (learned)', density=True)
        ax2.hist(w_u1_flat, bins=50, alpha=0.7, label='w_u (ground truth)', density=True)
        ax2.set_title('Unique Distribution')
        ax2.legend()
        ax2.set_xlabel('Value')
        
        ax3 = fig.add_subplot(3, 3, 3)
        ax3.hist(z_s_flat, bins=50, alpha=0.7, label='z_s (learned)', density=True)
        ax3.hist(w_s_flat, bins=50, alpha=0.7, label='w_s (ground truth)', density=True)
        ax3.set_title('Synergy Distribution')
        ax3.legend()
        ax3.set_xlabel('Value')
        
        # 2. Scatter plots: learned vs ground truth (middle row)
        ax4 = fig.add_subplot(3, 3, 4)
        ax4.scatter(w_r_flat, z_r_flat, alpha=0.3, s=5)
        ax4.set_xlabel('Ground Truth (w_r)')
        ax4.set_ylabel('Learned (z_r)')
        ax4.set_title(f'Redundancy Correlation: {np.corrcoef(w_r_flat, z_r_flat)[0,1]:.3f}')
        # Add identity line
        lims = [min(ax4.get_xlim()[0], ax4.get_ylim()[0]), max(ax4.get_xlim()[1], ax4.get_ylim()[1])]
        ax4.plot(lims, lims, 'r--', alpha=0.5)
        
        ax5 = fig.add_subplot(3, 3, 5)
        ax5.scatter(w_u1_flat, z_u1_flat, alpha=0.3, s=5)
        ax5.set_xlabel('Ground Truth (w_u)')
        ax5.set_ylabel('Learned (z_u)')
        ax5.set_title(f'Unique Correlation: {np.corrcoef(w_u1_flat, z_u1_flat)[0,1]:.3f}')
        lims = [min(ax5.get_xlim()[0], ax5.get_ylim()[0]), max(ax5.get_xlim()[1], ax5.get_ylim()[1])]
        ax5.plot(lims, lims, 'r--', alpha=0.5)
        
        ax6 = fig.add_subplot(3, 3, 6)
        ax6.scatter(w_s_flat, z_s_flat, alpha=0.3, s=5)
        ax6.set_xlabel('Ground Truth (w_s)')
        ax6.set_ylabel('Learned (z_s)')
        ax6.set_title(f'Synergy Correlation: {np.corrcoef(w_s_flat, z_s_flat)[0,1]:.3f}')
        lims = [min(ax6.get_xlim()[0], ax6.get_ylim()[0]), max(ax6.get_xlim()[1], ax6.get_ylim()[1])]
        ax6.plot(lims, lims, 'r--', alpha=0.5)
        
        # 3. Cross-correlation matrix of learned tokens (bottom left)
        ax7 = fig.add_subplot(3, 3, 7)
        token_matrix = np.vstack([z_r_flat, z_u1_flat, flatten(z_u2), z_s_flat])
        corr_matrix = np.corrcoef(token_matrix)
        im = ax7.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        ax7.set_xticks([0, 1, 2, 3])
        ax7.set_xticklabels(['z_r', 'z_u1', 'z_u2', 'z_s'])
        ax7.set_yticks([0, 1, 2, 3])
        ax7.set_yticklabels(['z_r', 'z_u1', 'z_u2', 'z_s'])
        ax7.set_title('Token Cross-Correlation (Orthogonality)')
        plt.colorbar(im, ax=ax7)
        # Annotate values
        for i in range(4):
            for j in range(4):
                ax7.text(j, i, f'{corr_matrix[i,j]:.2f}', ha='center', va='center',
                        color='white' if abs(corr_matrix[i,j]) > 0.5 else 'black')
        
        # 4. Summary metrics (bottom middle)
        ax8 = fig.add_subplot(3, 3, 8)
        ax8.axis('off')
        
        # Compute summary stats
        corr_r = np.corrcoef(w_r_flat, z_r_flat)[0, 1]
        corr_u = np.corrcoef(w_u1_flat, z_u1_flat)[0, 1]
        corr_s = np.corrcoef(w_s_flat, z_s_flat)[0, 1]
        
        off_diag = [corr_matrix[0,1], corr_matrix[0,2], corr_matrix[0,3],
                    corr_matrix[1,2], corr_matrix[1,3], corr_matrix[2,3]]
        mean_orth = np.mean(np.abs(off_diag))
        
        summary_text = f"""
PID Component Recovery Summary
{'='*40}

Redundancy (z_r ↔ w_r):  {corr_r:+.4f}  {'✓' if corr_r > 0.5 else '✗'}
Unique (z_u ↔ w_u):      {corr_u:+.4f}  {'✓' if corr_u > 0.4 else '✗'}
Synergy (z_s ↔ w_s):     {corr_s:+.4f}  {'✓' if corr_s > 0.3 else '✗'}

Token Orthogonality:
  Mean |off-diagonal|:   {mean_orth:.4f}  {'✓' if mean_orth < 0.2 else '✗'}

Token Statistics:
  std(z_r): {np.std(z_r_flat):.4f}
  std(z_u): {np.std(z_u1_flat):.4f}
  std(z_s): {np.std(z_s_flat):.4f}
"""
        ax8.text(0.1, 0.9, summary_text, transform=ax8.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        # 5. Token variance over samples (bottom right)
        ax9 = fig.add_subplot(3, 3, 9)
        sample_indices = np.arange(min(100, len(z_r_flat)))
        ax9.plot(sample_indices, z_r_flat[:100], label='z_r', alpha=0.7)
        ax9.plot(sample_indices, z_u1_flat[:100], label='z_u', alpha=0.7)
        ax9.plot(sample_indices, z_s_flat[:100], label='z_s', alpha=0.7)
        ax9.set_xlabel('Sample Index')
        ax9.set_ylabel('Token Value (mean)')
        ax9.set_title('Token Values Across Samples')
        ax9.legend()
        ax9.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig.savefig(self.figures_dir / "pid_analysis.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"[ExperimentLogger] PID analysis saved to {self.figures_dir / 'pid_analysis.png'}")
    
    def get_config(self) -> Dict[str, Any]:
        """Return the merged configuration."""
        return self.config
    
    def get_metrics_history(self) -> List[Dict[str, Any]]:
        """
        Return metrics history in format expected by TokenizerVisualizer.
        
        Transforms logged epoch data into visualization-compatible format.
        """
        history = []
        for epoch_data in self.metrics.get("epochs", []):
            entry = {
                "epoch": epoch_data.get("epoch", 0),
                "reconstruction_mse": epoch_data.get("loss_breakdown", {}).get("reconstruction", 
                                      epoch_data.get("train_loss", 0)),
                "val_reconstruction_mse": epoch_data.get("val_loss", 0),
            }
            # Add metrics
            metrics = epoch_data.get("metrics", {})
            entry["perplexity"] = metrics.get("perplexity", 0)
            entry["code_utilization"] = metrics.get("train_utilization", 
                                        metrics.get("val_utilization", 0))
            entry["dead_codes"] = entry.get("code_utilization", 1) * self.config.get("model", {}).get(
                "quantizer", {}).get("codebook_size", 1024)
            entry["dead_codes"] = int((1 - entry["code_utilization"]) * self.config.get("model", {}).get(
                "quantizer", {}).get("codebook_size", 1024))
            
            # Add loss breakdown
            for key, val in epoch_data.get("loss_breakdown", {}).items():
                entry[f"{key}_mse"] = val
            
            history.append(entry)
        
        return history


def _iter_metric_run_dirs(runs_dir: Path) -> Iterator[Path]:
    """Yield non-archived run directories that contain a metrics.json file."""
    if not runs_dir.exists():
        return

    for metrics_path in sorted(runs_dir.rglob("metrics.json")):
        run_dir = metrics_path.parent
        try:
            relative_parts = run_dir.relative_to(runs_dir).parts
        except ValueError:
            continue
        if "archive" in relative_parts:
            continue
        if (run_dir / "config.yaml").exists():
            yield run_dir


def update_comparison_csv(experiments_dir: str = "experiments"):
    """Aggregate all experiment results into a comparison CSV."""
    import csv
    
    experiments_dir = Path(experiments_dir)
    runs_dir = experiments_dir / "runs"
    results_dir = experiments_dir / "results"
    results_dir.mkdir(exist_ok=True)
    
    csv_path = results_dir / "comparison.csv"
    
    rows = []
    for run_dir in _iter_metric_run_dirs(runs_dir):
        metrics_path = run_dir / "metrics.json"
        config_path = run_dir / "config.yaml"
        
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        final = metrics.get("final_metrics", {})
        loss_config = config.get("loss", {})
        
        row = {
            "exp_id": config.get("experiment", {}).get("name", "unknown"),
            "run_name": run_dir.name,
            "config": f"{loss_config.get('alignment', {}).get('type', 'N/A')}+"
                      f"{loss_config.get('orthogonality', {}).get('type', 'N/A')}+"
                      f"{loss_config.get('synergy', {}).get('type', 'N/A')}",
            "corr_zr": final.get("corr_zr_wr", ""),
            "corr_zu": final.get("corr_zu_wu", ""),
            "corr_zs": final.get("corr_zs_ws", ""),
            "hsic": final.get("hsic_zr_zu", ""),
            "completed": metrics.get("completed_at", "")
        }
        rows.append(row)
    
    if rows:
        fieldnames = ["exp_id", "run_name", "config", "corr_zr", "corr_zu", "corr_zs", "hsic", "completed"]
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        
        print(f"[Results] Updated comparison CSV: {csv_path}")
        print(f"[Results] Total experiments: {len(rows)}")

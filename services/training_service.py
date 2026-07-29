import os
import sys
import time
import signal
import shutil
import json
import tempfile
import subprocess
import threading
import traceback
from pathlib import Path
from typing import Dict, Tuple

import yaml

from gui.qt_compat import QObject, pyqtSignal

class TrainingWorkerSignals(QObject):
    log_signal = pyqtSignal(str)
    event_signal = pyqtSignal(str, dict)
    progress_signal = pyqtSignal(int, int)  # current_epoch, total_epochs

def _normalize_training_data_yaml(train_args: Dict) -> Dict:
    normalized_args = dict(train_args)
    raw_yaml = normalized_args.get("data") or normalized_args.get("data_yaml")
    if not raw_yaml:
        return normalized_args

    yaml_path = Path(raw_yaml).expanduser()
    if not yaml_path.is_absolute():
        yaml_path = (Path(__file__).resolve().parent.parent / yaml_path).resolve()
    else:
        yaml_path = yaml_path.resolve()

    if not yaml_path.exists():
        normalized_args["data"] = str(yaml_path)
        normalized_args["data_yaml"] = str(yaml_path)
        return normalized_args

    with open(yaml_path, "r", encoding="utf-8") as f:
        yaml_data = yaml.safe_load(f) or {}

    dataset_root = yaml_data.get("path")
    if dataset_root:
        dataset_root_path = Path(dataset_root).expanduser()
        if not dataset_root_path.is_absolute():
            candidate_from_yaml = (yaml_path.parent / dataset_root_path).resolve()
            candidate_from_project = (Path(__file__).resolve().parent.parent / dataset_root_path).resolve()

            if (candidate_from_yaml / "train").exists() or (candidate_from_yaml / "images").exists():
                dataset_root_path = candidate_from_yaml
            else:
                dataset_root_path = candidate_from_project
        else:
            dataset_root_path = dataset_root_path.resolve()
        yaml_data["path"] = str(dataset_root_path)

        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(yaml_data, f, allow_unicode=True, sort_keys=False)

    normalized_args["data"] = str(yaml_path)
    normalized_args["data_yaml"] = str(yaml_path)
    return normalized_args


class TrainingManager(QObject):
    def __init__(self):
        super().__init__()
        self.signals = TrainingWorkerSignals()
        self.is_training = False
        self.stop_requested = False
        self.training_process = None
        self._thread = None

    def start_training(self, train_args: Dict) -> Tuple[bool, str]:
        if self.is_training:
            return False, "Training is already in progress"

        train_args = _normalize_training_data_yaml(train_args)
        self.is_training = True
        self.stop_requested = False

        self._thread = threading.Thread(target=self._run_training_process, args=(train_args,), daemon=True)
        self._thread.start()
        return True, "Training started"

    def stop_training(self) -> bool:
        if not self.is_training:
            return False
        
        self.stop_requested = True
        self.signals.log_signal.emit("[SYSTEM] Termination request received. Force killing training process...")
        
        if self.training_process and self.training_process.poll() is None:
            self._kill_process_tree(self.training_process)
            
        self.is_training = False
        self.signals.event_signal.emit("training_stopped", {})
        self.signals.log_signal.emit("[SYSTEM] Training process killed successfully!")
        return True

    def _check_and_emit_epoch_progress(self, line: str, target_total_epochs: int = None):
        if not line:
            return
        if "[EPOCH_PROGRESS]" in line:
            try:
                parts = line.split("[EPOCH_PROGRESS]")[1].strip().split("/")
                cur = int(parts[0])
                tot = int(parts[1])
                self.signals.progress_signal.emit(cur, tot)
                self.signals.event_signal.emit("epoch_end", {"epoch": cur, "total_epochs": tot})
            except Exception:
                pass
        else:
            # 过滤掉 Validation/Batch 级别的子进度 (例如 1/7, 35/36 等)
            import re
            match = re.search(r'^\s*(\d+)/(\d+)\b', line)
            if match:
                try:
                    cur = int(match.group(1))
                    tot = int(match.group(2))
                    if target_total_epochs and tot != target_total_epochs:
                        return
                    if 0 <= cur <= tot and tot > 0:
                        self.signals.progress_signal.emit(cur, tot)
                        self.signals.event_signal.emit("epoch_end", {"epoch": cur, "total_epochs": tot})
                except Exception:
                    pass

    def _kill_process_tree(self, proc):
        if proc is None:
            return
        try:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
            else:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception:
                    proc.kill()
        except Exception as e:
            try:
                proc.kill()
            except Exception:
                pass

    def _run_training_process(self, train_args: Dict):
        temp_script = None
        try:
            total_epochs = train_args.get("epochs", 1000)
            self.signals.event_signal.emit("training_started", {"total_epochs": total_epochs})
            self.signals.log_signal.emit(f"=== Starting Ultralytics Model Training ===")
            self.signals.log_signal.emit(f"Model: {train_args.get('model')}")
            self.signals.log_signal.emit(f"Dataset: {train_args.get('data')}")
            self.signals.log_signal.emit(f"Epochs: {total_epochs}, Batch Size: {train_args.get('batch_size')}, ImgSz: {train_args.get('imgsz')}")

            fd, temp_script = tempfile.mkstemp(prefix="yolo_worker_", suffix=".py")
            
            project_root = str(Path(__file__).resolve().parent.parent)
            payload_json_str = json.dumps(train_args, ensure_ascii=False)

            script_content = f"""import sys
import os
import json
import traceback
from pathlib import Path

sys.path.insert(0, r"{project_root}")

from services.config import init_pretrained_model_env
init_pretrained_model_env()

from utils.train_utils import YOLOTrainer

def run():
    raw_json = {json.dumps(payload_json_str)}
    train_args = json.loads(raw_json)
    
    model_type = train_args.get("model", "yolo11s-seg.pt")
    task = train_args.get("task", "segment")
    data_yaml = train_args.get("data")
    device = train_args.get("device", "0")
    iteration_path = train_args.get("iteration_path", None)

    project_root_path = Path(r"{project_root}")
    pretrained_dir = project_root_path / "pre_trained_model"

    if not iteration_path:
        if os.path.exists(model_type):
            iteration_path = model_type
        elif (pretrained_dir / model_type).exists():
            iteration_path = str(pretrained_dir / model_type)
        elif (project_root_path / model_type).exists():
            iteration_path = str(project_root_path / model_type)

    model_stem = Path(model_type).stem
    yolo_version = train_args.get("yolo_version", "yolo11")

    trainer = YOLOTrainer(
        model_type=model_stem if not iteration_path else model_type,
        task=task,
        yolo_version=yolo_version,
        iteration_path=iteration_path
    )

    clean_kwargs = dict(train_args)
    for k in ["model", "model_type", "data", "task", "yolo_version", "iteration_path", "data_yaml"]:
        clean_kwargs.pop(k, None)

    trainer.train(
        data_yaml=data_yaml,
        **clean_kwargs
    )

if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        print(f"[Worker Error] {{e}}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
"""
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(script_content)

            creationflags = 0
            if os.name == "nt" and hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):
                creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            env["PYTHONIOENCODING"] = "utf-8"

            self.training_process = subprocess.Popen(
                [sys.executable, temp_script],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=env,
                preexec_fn=os.setsid if os.name != "nt" else None,
                creationflags=creationflags,
            )

            import re

            # Read characters splitting on both \r and \n to capture live TQDM progress lines!
            buffer = ""
            while True:
                if self.stop_requested:
                    self._kill_process_tree(self.training_process)
                    self.is_training = False
                    return

                char = self.training_process.stdout.read(1)
                if not char and self.training_process.poll() is not None:
                    if buffer.strip():
                        line = buffer.strip()
                        self.signals.log_signal.emit(line)
                        self._check_and_emit_epoch_progress(line, total_epochs)
                    break

                if char in ('\r', '\n'):
                    if buffer.strip():
                        line = buffer.strip()
                        self.signals.log_signal.emit(line)
                        self._check_and_emit_epoch_progress(line, total_epochs)
                    buffer = ""
                else:
                    buffer += char

            return_code = self.training_process.poll()
            self.is_training = False

            if self.stop_requested:
                self.signals.event_signal.emit("training_stopped", {})
            elif return_code == 0:
                self.signals.event_signal.emit("training_completed", {"results": "Training completed successfully"})
                self.signals.log_signal.emit("[SYSTEM] Training completed successfully!")
            else:
                self.signals.event_signal.emit("training_error", {"error": f"Process exited with code: {return_code}"})
                self.signals.log_signal.emit(f"[ERROR] Process exited with non-zero exit code: {return_code}")

        except Exception as e:
            self.is_training = False
            err_msg = f"Failed to launch training process: {str(e)}"
            self.signals.log_signal.emit(f"[ERROR] {err_msg}")
            self.signals.event_signal.emit("training_error", {"error": err_msg})
        finally:
            if temp_script and os.path.exists(temp_script):
                try:
                    os.remove(temp_script)
                except OSError:
                    pass

_training_manager = TrainingManager()

def get_training_manager() -> TrainingManager:
    return _training_manager

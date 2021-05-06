from pathlib import Path
from typing import Dict, Optional
from typing import Tuple


def find_latest_model(root_path: Path) -> Optional[Path]:
    import os
    from pathlib import Path
    logs_path = (root_path / "checkpoints")
    if logs_path.exists() is False:
        print(f"No previous record found in {logs_path}")
        return None
    paths = sorted(logs_path.iterdir(), key=os.path.getmtime)
    paths_dict: Dict[int, Path] = {
        int(path.name.split("_")[2]): path for path in paths
    }
    if len(paths_dict) == 0:
        return None
    return paths[-1]
    
    latest_model_file_path: Optional[Path] = paths_dict.get(max(paths_dict.keys()), None)
    return latest_model_file_path


def prep_dir(output_folder_path: Path) -> Tuple[Path, Path]:
    tensorboard_dir = (output_folder_path / "tensorboard")
    ckpt_dir = (output_folder_path / "checkpoints")
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return tensorboard_dir, ckpt_dir

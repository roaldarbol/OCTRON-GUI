"""
OCTRON training pipeline.

Wraps the YOLO_octron model-loading and training steps into a single callable.
By default, training data is prepared automatically via ``run_split()``.
Pass ``skip_split=True`` if ``octron split`` has already been run.

External training data (not from an OCTRON project) can be used by passing
``data_dir`` pointing to a directory that contains a valid ``yolo_config.yaml``
together with the standard YOLO train/val/test split subdirectories.
"""

import json
from pathlib import Path
from typing import Optional

_MODELS_YAML = Path(__file__).parent.parent / "yolo_octron" / "yolo_models.yaml"


def _get_batch_size(model, imgsz, device, cache_path):
    """
    Return the optimal batch size for training, using a cache to avoid
    running AutoBatch on every training run.

    The cache is stored at ``cache_path`` and keyed by model architecture,
    image size, and GPU name so it stays valid across runs with the same
    hardware and model.
    """
    import torch

    gpu_name = (
        torch.cuda.get_device_name(device) if torch.cuda.is_available() else "cpu"
    )
    cache_key = f"{model.model.model.__class__.__name__}_{imgsz}_{gpu_name}"

    cache_path = Path(cache_path)
    cache = {}
    if cache_path.exists():
        try:
            with open(cache_path) as f:
                cache = json.load(f)
        except Exception:
            cache = {}

    if cache_key in cache:
        batch = cache[cache_key]
        print(f"AutoBatch: using cached batch size {batch} for {gpu_name} (imgsz={imgsz})")
        return batch

    print("AutoBatch: running batch size search (result will be cached) ...")
    from ultralytics.utils.autobatch import check_train_batch_size
    batch = check_train_batch_size(model.model.model, imgsz=imgsz, amp=True)

    cache[cache_key] = batch
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(cache, f)
    print(f"AutoBatch: batch size {batch} cached to {cache_path}")
    return batch


def _normalise_model_name(model, models_yaml_path):
    """Return the canonical model key from models_yaml (case-insensitive match)."""
    import yaml
    model_str = model.value if hasattr(model, 'value') else str(model)
    with open(models_yaml_path) as f:
        models_dict = yaml.safe_load(f)
    match = next((k for k in models_dict if k.lower() == model_str.lower()), None)
    return match if match is not None else model_str


def _validate_data_dir(data_dir: Path):
    """
    Validate that *data_dir* contains a complete YOLO training dataset.

    Checks performed:
    - ``yolo_config.yaml`` exists and is parseable
    - Required fields ``names``, ``train``, and ``val`` are present
    - Train and val subdirectories exist
    - At least one label file (.txt) is present in the train split

    Returns
    -------
    cfg : dict
        Parsed contents of ``yolo_config.yaml``.
    train_mode : str or None
        Value of ``train_mode`` from the yaml, or ``None`` if not present.
    """
    import yaml

    config_path = data_dir / "yolo_config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"No yolo_config.yaml found in {data_dir}. "
            "The data directory must contain a valid YOLO config file."
        )

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    for field in ("names", "train", "val"):
        if field not in cfg:
            raise ValueError(
                f"yolo_config.yaml is missing required field '{field}'."
            )

    train_subdir = data_dir / cfg["train"]
    val_subdir = data_dir / cfg["val"]

    if not train_subdir.exists():
        raise FileNotFoundError(
            f"Training split directory not found: {train_subdir}"
        )
    if not val_subdir.exists():
        raise FileNotFoundError(
            f"Validation split directory not found: {val_subdir}"
        )

    label_files = list(train_subdir.glob("*.txt"))
    if not label_files:
        raise FileNotFoundError(
            f"No label files (.txt) found in training split: {train_subdir}"
        )

    yaml_mode = cfg.get("train_mode")
    return cfg, yaml_mode


def run_training(
    # --- Data source ---
    project_path=None,
    data_dir: Optional[Path] = None,
    # --- Model ---
    model="YOLO26m",
    train_mode=None,
    # --- Core hyperparameters ---
    epochs=250,
    imagesz=640,
    device="auto",
    save_period=50,
    # --- Output control ---
    output_dir: Optional[Path] = None,
    run_name: Optional[str] = None,
    overwrite=False,
    resume=False,
    # --- Data split (OCTRON projects only) ---
    skip_split=False,
    train_fraction=0.7,
    val_fraction=0.15,
    seed=88,
):
    """
    Run the OCTRON/YOLO training pipeline.

    By default this prepares and exports training data before training.
    Pass ``skip_split=True`` to skip that step when data is already up to date.

    External training data (not from an OCTRON project) can be used by passing
    ``data_dir`` pointing to a directory with a ``yolo_config.yaml`` file and
    the corresponding YOLO split subdirectories.

    Parameters
    ----------
    project_path : str or Path, optional
        Path to the OCTRON project directory. Required unless ``data_dir`` is given.
    data_dir : Path, optional
        Path to an external YOLO training data directory. When provided,
        ``skip_split`` is implied and ``train_mode`` is read from
        ``yolo_config.yaml`` if not explicitly supplied.
    model : str or Path
        YOLO model name (e.g. 'YOLO11m') or path to an existing model file.
    train_mode : str or None
        'segment' or 'detect'. When ``None`` and ``data_dir`` is provided the
        value is read from ``yolo_config.yaml``; falls back to 'segment'.
    epochs : int
        Number of training epochs.
    imagesz : int
        Input image size for training.
    device : str
        Device to train on ('auto', 'cpu', 'cuda', 'mps'). 'auto' selects
        CUDA if available, then MPS, then CPU.
    save_period : int
        Save a checkpoint every N epochs.
    output_dir : Path, optional
        Base directory where the training run folder will be created.
        Defaults to ``<project_path>/model/`` or ``<data_dir>/model/``.
    run_name : str, optional
        Name of the training run subdirectory. When ``None`` the GUI default
        ``'training'`` is used, preserving existing behaviour. CLI callers
        should pass an informative name such as ``'yolo26m_seg_640_20260327'``.
    overwrite : bool
        Overwrite an existing trained model. Default: skip if best.pt exists.
    resume : bool
        Resume training from an existing last.pt checkpoint.
    skip_split : bool
        Skip data preparation. Use when ``octron split`` has already been run
        and the training data is up to date.
    train_fraction : float
        Fraction of frames for training (ignored when ``skip_split=True``).
    val_fraction : float
        Fraction of frames for validation (ignored when ``skip_split=True``).
    seed : int
        Random seed for the split (ignored when ``skip_split=True``).
    """
    from octron.yolo_octron.yolo_octron import YOLO_octron
    from octron.test_gpu import auto_device
    from octron.tools.split import run_split

    # Unwrap enums to plain strings
    device = device.value if hasattr(device, 'value') else str(device)
    if train_mode is not None:
        train_mode = train_mode.value if hasattr(train_mode, 'value') else str(train_mode)

    # --- Resolve paths and config based on data source ---
    if data_dir is not None:
        data_dir = Path(data_dir)
        cfg, yaml_mode = _validate_data_dir(data_dir)
        # Use yaml train_mode if caller didn't specify one explicitly
        if train_mode is None:
            train_mode = yaml_mode
        config_path = data_dir / "yolo_config.yaml"
        img_search_path = data_dir
        output_base = Path(output_dir) if output_dir else data_dir / "model"
        batch_cache = output_base / "autobatch_cache.json"
        skip_split = True  # external data is already prepared
    else:
        if project_path is None:
            raise ValueError("Either project_path or data_dir must be provided.")
        project_path = Path(project_path)
        config_path = project_path / "model" / "training_data" / "yolo_config.yaml"
        img_search_path = project_path / "model" / "training_data"
        output_base = Path(output_dir) if output_dir else project_path / "model"
        batch_cache = project_path / "model" / "autobatch_cache.json"

    # Default train_mode if still unresolved
    if train_mode is None:
        train_mode = "segment"

    # Default run_name: GUI always passes 'training'; CLI passes an informative name
    if run_name is None:
        run_name = "training"

    # --- Check for existing run ---
    best_pt = output_base / run_name / "weights" / "best.pt"
    last_pt = output_base / run_name / "weights" / "last.pt"
    if resume:
        if best_pt.exists():
            print(f"Training already completed ({best_pt}). Nothing to resume. Use --overwrite to retrain from scratch.")
            return
        if not last_pt.exists():
            print("No interrupted training found (last.pt missing). Starting fresh.")
            resume = False
    elif best_pt.exists() and not overwrite:
        print(f"Trained model already exists at {best_pt}. Use --overwrite to retrain.")
        return

    if device == "auto":
        device = auto_device()

    # --- Steps 1–4: prepare and export training data (OCTRON projects only) ---
    # Skip the expensive JSON-loading pipeline if training data is already on
    # disk and the caller hasn't asked to overwrite it.  Preparing labels reads
    # every object_organizer.json in the project, which can take a long time on
    # large projects.  If the yolo_config.yaml is already present the data is
    # ready to use; pass --overwrite to force regeneration.
    if not skip_split and config_path.exists() and not overwrite:
        print(
            f"Training data already exists at {config_path.parent} — skipping data "
            "preparation. Delete the training_data folder or use --overwrite to regenerate."
        )
        skip_split = True

    if not skip_split:
        run_split(
            project_path=project_path,
            train_fraction=train_fraction,
            val_fraction=val_fraction,
            seed=seed,
            train_mode=train_mode,
            dry_run=False,
        )

    # --- Step 5: load the base model (or last.pt when resuming) ---
    # When only --data-dir is provided, output_base (data_dir/model/) is used
    # as the YOLO_octron project root. Create it if needed so the path-existence
    # check in YOLO_octron's project_path setter doesn't raise.
    yolo_project_path = project_path if project_path is not None else output_base
    yolo_project_path.mkdir(parents=True, exist_ok=True)
    yolo = YOLO_octron(
        models_yaml_path=_MODELS_YAML,
        project_path=yolo_project_path,
        clean_training_dir=False,
    )
    yolo.train_mode = train_mode
    yolo.config_path = config_path
    yolo.data_path = img_search_path
    yolo.training_path = output_base

    if resume:
        print(f"Resuming from checkpoint: {last_pt}")
        yolo.load_model(last_pt, train_mode=train_mode)
    else:
        model = _normalise_model_name(model, _MODELS_YAML)
        print(f"Loading model: {model}...")
        yolo.load_model(model, train_mode=train_mode)

    # --- Step 6: train ---
    batch = _get_batch_size(yolo, imagesz, device, batch_cache)

    print(f"Training for {epochs} epochs on {device}...")
    print(f"Run name: {run_name}")
    print(f"Output:   {output_base / run_name}")
    for progress in yolo.train(
        device=device,
        imagesz=imagesz,
        epochs=epochs,
        save_period=save_period,
        train_mode=train_mode,
        resume=resume,
        batch=batch,
        run_name=run_name,
    ):
        epoch = progress.get("epoch", "?")
        total_epochs = progress.get("total_epochs", "?")
        remaining = progress.get("remaining_time", 0)
        print(f"  Epoch {epoch}/{total_epochs} | ETA: {remaining:.0f}s", end="\r")
    print()
    print("Training complete.")

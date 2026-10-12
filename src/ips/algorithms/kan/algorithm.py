"""KAN-Mixer IPS lifecycle; torch imports stay inside this backend."""
import csv
import json
from pathlib import Path

from .paths import MODEL_CONFIG

from ..base import Algorithm
from ...data import ROOT, sha256
from ...data_access import FingerprintInputs, FingerprintSplit
from ...experiments.contracts import AlgorithmExecution, ExperimentConfig, IPSPredictions, ModelDiagram


class KANMixerAlgorithm(Algorithm):
    """IPS KAN-Mixer using efficient-kan B-splines, directly implementing Algorithm."""
    model_id = "kan_mixer"

    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        unknown = set(config.params) - {"model_config", "epochs", "checkpoint"}
        if unknown:
            raise ValueError(f"Unknown KAN-Mixer params: {sorted(unknown)}")
        if "checkpoint" in config.params and "epochs" in config.params:
            raise ValueError("Checkpoint evaluation cannot also request training epochs.")
        epochs = config.params.get("epochs")
        if epochs is not None and (type(epochs) is not int or epochs < 1):
            raise ValueError("KAN-Mixer epochs must be a positive integer.")
        self._train = self._tensor_train = self._model = self._checkpoint = None

    def prepare(self, train: FingerprintSplit) -> None:
        if train.name != "train":
            raise ValueError("KAN-Mixer preparation requires the train split.")
        self._train = train
        if "checkpoint" not in self.config.params:
            from .dataset import FingerprintDataset
            from .runtime import read_config
            path = Path(self.config.params.get("model_config", MODEL_CONFIG))
            path = path if path.is_absolute() else ROOT/path
            self._model_config = read_config(path)
            self._model_config["model"] = self.model_id
            # The Runner's audited dataset is authoritative, not a second config's split.
            self._model_config["split_dir"] = str(train.info.directory)
            self._tensor_train = FingerprintDataset.from_split(train)

    def execute(self, validation: FingerprintSplit | None) -> AlgorithmExecution:
        if self._train is None:
            raise RuntimeError("Call prepare(train) before execute().")
        from .runtime import configure_runtime, load_checkpoint
        output = self.config.output_dir/"algorithm"
        artifacts, history, resources = {}, None, None
        if "checkpoint" in self.config.params:
            path = Path(self.config.params["checkpoint"])
            path = path if path.is_absolute() else ROOT/path
        else:
            from .dataset import FingerprintDataset
            from .train import train
            if validation is None or validation.name != "validation":
                raise ValueError("KAN-Mixer training requires internal validation, never test.")
            if (validation.info != self._train.info or
                    set(validation.inputs.row_indices) & set(self._train.inputs.row_indices)):
                raise ValueError("Training/validation provenance differs or rows overlap.")
            train(self._model_config, output, self.config.seed, self.config.device,
                self.config.params.get("epochs"), self.config.smoke, bundle=self._train.info,
                train_data=self._tensor_train, validation_data=FingerprintDataset.from_split(validation))
            path = output/"best.pt"
            for name in ("best.pt", "last.pt", "history.csv", "run_metadata.json", "resources.json", "kan_diagnostics.json"):
                if (output/name).exists():
                    artifacts[f"kan_{name}"] = output/name
            with (output/"history.csv").open(encoding="utf-8", newline="") as stream:
                history = [dict(row) for row in csv.DictReader(stream)]
            resources = json.loads((output/"resources.json").read_text(encoding="utf-8"))
            resources["factory_scope_note"] = (
                "Legacy neural execution with already-prepared data; backend pipeline_seconds excludes "
                "shared repository audit, tensor preparation, prediction and export. See RunResult.runtime.")
        device = configure_runtime(self.config.seed, self.config.device)
        self._device = device
        self._model, self._checkpoint = load_checkpoint(path, device)
        checkpoint, info = self._checkpoint, self._train.info
        if checkpoint.get("metadata", {}).get("full_development_refit") and "validation" in self.config.evaluation_splits:
            raise ValueError("Final refit incorporated validation into training; request test-only evaluation.")
        if (checkpoint["config"]["model"] != self.model_id or checkpoint["ap_order"] != info.ap_order or
                checkpoint["classes"] != info.classes or checkpoint["preprocessing"] != info.preprocessing or
                checkpoint["split_manifest_sha256"] != sha256(info.directory/"manifest.json")):
            raise ValueError("Checkpoint algorithm, split manifest or preprocessing/mapping differ.")
        artifacts["checkpoint"] = path.resolve()
        finalized = checkpoint["run_type"] == "training"
        run_type = checkpoint["run_type"]
        if "checkpoint" in self.config.params:
            run_type = "checkpoint_evaluation" if finalized else "smoke_checkpoint_evaluation"
        return AlgorithmExecution(finalized, run_type, history, artifacts, resources)

    def predict(self, inputs: FingerprintInputs) -> IPSPredictions:
        if self._model is None:
            raise RuntimeError("Call execute() to train/load a selected checkpoint before predict().")
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from ...data import feature_transform
        from ...metrics import inverse_xy
        from .train import infer
        features = torch.from_numpy(feature_transform(inputs.rssi))
        loader = DataLoader(TensorDataset(features, torch.zeros(len(inputs), dtype=torch.long),
            torch.zeros(len(inputs), 2)), batch_size=64, shuffle=False, num_workers=0)
        logits, xy, _ = infer(self._model, loader, self._device)
        probabilities, classes = logits.softmax(dim=1).max(dim=1)
        return IPSPredictions(inputs.row_indices.copy(), inputs.source_id,
            xy=inverse_xy(xy.numpy(), self._checkpoint["preprocessing"]),
            class_ids=classes.numpy(), confidence=probabilities.numpy())

    def get_diagram(self) -> ModelDiagram | None:
        if self._checkpoint is None:
            return None
        settings = self._checkpoint["config"]
        mixer = settings["mixer"]
        backbone = (f"efficient-kan B-spline: {mixer['tokens']} tokens, {mixer['channels']} channels, "
                    f"{mixer['blocks']} blocks")
        if settings.get("positioning", {}).get("variant") == "reference":
            backbone += f"; {len(settings['positioning']['references'])} train reference points"
        if "ensemble_members" in self._checkpoint:
            backbone += f"; ensemble of {len(self._checkpoint['ensemble_members'])} KAN-Mixers"
        if "fingerprint_refinement" in self._checkpoint:
            backbone += "; explicit hybrid with train-only fingerprint retrieval"
        return ModelDiagram('flowchart LR\n    A["520 AP: RSSI + mask"] --> B["'+backbone+
            f'"]\n    B --> C["{len(self._checkpoint["classes"])} logits: building/floor"]'
            '\n    B --> D["2 coordinates: inverse to metres"]\n', "mermaid")

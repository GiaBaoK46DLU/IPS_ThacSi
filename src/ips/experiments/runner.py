"""Algorithm lifecycle and audited shared splits, without a neural training assumption."""
import contextlib
import sys
import time

from ..data import ROOT, sha256
from ..data_access import DatasetRepository
from .contracts import AlgorithmExecution, ExperimentConfig, RunResult
from .evaluator import IPSEvaluator
from .exporter import ReportExporter, save_json
from .factory import default_factory


class _Tee:
    def __init__(self, terminal, log):
        self.terminal, self.log = terminal, log

    def write(self, text):
        self.terminal.write(text)
        self.log.write(text)
        self.flush()
        return len(text)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


class ExperimentRunner:
    def __init__(self, factory=None, repository=None, evaluator=None, exporter=None):
        self.factory = factory if factory is not None else default_factory()
        self.repository = repository if repository is not None else DatasetRepository()
        self.evaluator = evaluator if evaluator is not None else IPSEvaluator()
        self.exporter = exporter if exporter is not None else ReportExporter()

    def run(self, config: ExperimentConfig) -> RunResult:
        algorithm = self.factory.create(config.algorithm_id, config)
        # Never reuse a directory, even after a failed or interrupted run.
        config.output_dir.mkdir(parents=True, exist_ok=False)
        log_path = config.output_dir/"terminal.log"
        started = time.perf_counter()
        with log_path.open("w", encoding="utf-8") as log:
            with contextlib.redirect_stdout(_Tee(sys.stdout, log)), contextlib.redirect_stderr(_Tee(sys.stderr, log)):
                try:
                    print(f"Factory run: {config.algorithm_id}; seed={config.seed}; smoke={config.smoke}", flush=True)
                    runtime = {}
                    phase = time.perf_counter()
                    bundle = self.repository.load(config.split_dir)
                    train = bundle.split("train", 256 if config.smoke else None)
                    validation = bundle.split("validation", 128 if config.smoke else None)
                    runtime["data_seconds"] = time.perf_counter() - phase
                    phase = time.perf_counter()
                    algorithm.prepare(train)
                    runtime["prepare_seconds"] = time.perf_counter() - phase
                    phase = time.perf_counter()
                    execution = algorithm.execute(validation)
                    runtime["execute_seconds"] = time.perf_counter() - phase
                    if not isinstance(execution, AlgorithmExecution):
                        raise TypeError("execute() must return AlgorithmExecution.")
                    if "test" in config.evaluation_splits and not execution.finalized:
                        raise ValueError("Official test requires a finalized algorithm; smoke/unselected states are forbidden.")
                    predictions, metrics, matrices, splits = {}, {}, {}, {}
                    for name in config.evaluation_splits:
                        # Test features and labels are opened only after execute has frozen the state.
                        phase = time.perf_counter()
                        split = validation if name == "validation" else bundle.split("test")
                        if name == "test":
                            runtime["test_data_seconds"] = time.perf_counter() - phase
                        phase = time.perf_counter()
                        prediction = algorithm.predict(split.inputs)
                        runtime[name + "_predict_seconds"] = time.perf_counter() - phase
                        phase = time.perf_counter()
                        metrics[name], matrices[name] = self.evaluator.evaluate(split, prediction, config.task)
                        runtime[name + "_evaluate_seconds"] = time.perf_counter() - phase
                        predictions[name], splits[name] = prediction, split
                        print(f"{name}: n={len(split.inputs)}; mean_2d={metrics[name].get('mean_2d')}; "
                              f"joint_accuracy={metrics[name].get('joint_accuracy')}", flush=True)
                    result = RunResult(config.algorithm_id, predictions, metrics, matrices, runtime, log_path,
                                       config.snapshot(), "smoke_check" if config.smoke else execution.run_type,
                                       execution.history, dict(execution.artifacts), execution.resources,
                                       algorithm.get_diagram())
                    result.artifacts["terminal_log"] = log_path
                    provenance = {"split_manifest_sha256": sha256(config.split_dir/"manifest.json"),
                                  "split_manifest": bundle.manifest,
                                  "preprocessing": bundle.preprocessing, "classes": bundle.classes,
                                  "ap_order": bundle.ap_order,
                                  "code_sha256": {path.relative_to(ROOT).as_posix(): sha256(path)
                                      for path in sorted((ROOT/"src/ips").rglob("*.py"))},
                                  "preparation_scope": "train only; internal validation may select state; test after finalization"}
                    provenance_path = config.output_dir/"data_provenance.json"
                    save_json(provenance_path, provenance)
                    result.artifacts["data_provenance"] = provenance_path
                    phase = time.perf_counter()
                    result.artifacts.update(self.exporter.export(result, splits, config.output_dir, config.export_plots))
                    runtime["export_seconds"] = time.perf_counter() - phase
                    result.artifacts["run_result"] = config.output_dir/"run_result.json"
                    runtime["total_seconds"] = time.perf_counter() - started
                    save_json(result.artifacts["run_result"], result.summary())
                    print(f"Completed: {result.artifacts['run_result']}", flush=True)
                    return result
                except Exception as error:
                    print(f"FAILED: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
                    save_json(config.output_dir/"failure.json", {"error": type(error).__name__, "message": str(error),
                                                              "config": config.snapshot()})
                    raise

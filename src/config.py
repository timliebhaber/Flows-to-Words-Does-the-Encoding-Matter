"""Single entry point for loading YAML configs.

Code never defines parameter values itself: everything comes from the dicts returned here.
A missing key is an error (no fallback defaults).
"""

import math
import string
from pathlib import Path

import yaml

REQUIRED_DATA_KEYS = {
    "paths": ["raw_dir", "processed_dir", "split_file_suffix", "dataset_config_dir",
              "stats_dir", "stats_summary_file"],
    "seeds": ["sampling", "split"],
    "sampling": ["sample_size", "rare_class_threshold"],
    "cleaning": ["inf_to_missing"],
    "dedupe": ["conflict_policy", "min_rows_per_class"],
    "io": ["read_block_size_bytes"],
    "columns": ["label", "attack", "class", "row_id", "benign_class", "benign_label_value",
                "leaky", "features", "float_features", "string_features"],
}
REQUIRED_DATASET_KEYS = ["name", "raw_file", "label_map", "keep_all_classes", "string_value_map"]
CONFLICT_POLICIES = ["drop_all", "keep"]


class ConfigError(ValueError):
    pass


def _read_yaml(path):
    """YAML as a dict. A top-level `extends: <path>` starts from that file; each top-level section given here
    replaces the parent's section as a whole (no deep merge), so shared settings live in one place."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if isinstance(cfg, dict) and "extends" in cfg:
        child = {k: v for k, v in cfg.items() if k != "extends"}
        return {**_read_yaml(cfg["extends"]), **child}
    return cfg


def _require(mapping, keys, where):
    missing = [k for k in keys if k not in mapping]
    if missing:
        raise ConfigError(f"{where}: missing keys {missing}")


def validate_data_config(cfg):
    _require(cfg, list(REQUIRED_DATA_KEYS) + ["datasets", "split"], "data config")
    for section, keys in REQUIRED_DATA_KEYS.items():
        _require(cfg[section], keys, f"data config [{section}]")

    cols = cfg["columns"]
    features, leaky = cols["features"], cols["leaky"]
    if len(set(features)) != len(features):
        raise ConfigError("columns.features contains duplicates")
    overlap = set(features) & set(leaky)
    if overlap:
        raise ConfigError(f"leaky columns listed as features: {sorted(overlap)}")
    unknown_float = set(cols["float_features"]) - set(features)
    if unknown_float:
        raise ConfigError(f"float_features not in features: {sorted(unknown_float)}")
    strings = set(cols["string_features"])
    if strings - set(features) or strings & set(cols["float_features"]):
        raise ConfigError(f"string_features must be features and not float_features: {sorted(strings)}")
    added = [cols["class"], cols["row_id"]]
    raw_labels = [cols["label"], cols["attack"]]
    clash = set(added + raw_labels) & set(features + leaky)
    if clash:
        raise ConfigError(f"label/id column names clash with feature columns: {sorted(clash)}")

    ratios = list(cfg["split"].values())
    if len(ratios) != 3 or not math.isclose(sum(ratios), 1.0) or min(ratios) <= 0:
        raise ConfigError(f"split needs 3 positive ratios summing to 1, got {cfg['split']}")
    if cfg["dedupe"]["conflict_policy"] not in CONFLICT_POLICIES:
        raise ConfigError(f"dedupe.conflict_policy must be one of {CONFLICT_POLICIES}")


def validate_dataset_config(ds, data_cfg, name):
    _require(ds, REQUIRED_DATASET_KEYS, f"dataset config '{name}'")
    canonical = set(ds["label_map"].values())
    benign = data_cfg["columns"]["benign_class"]
    if benign not in canonical:
        raise ConfigError(f"dataset '{name}': label_map has no '{benign}' class")
    unknown_keep = set(ds["keep_all_classes"]) - canonical
    if unknown_keep:
        raise ConfigError(f"dataset '{name}': keep_all_classes not in label_map values: {sorted(unknown_keep)}")
    if ds["string_value_map"] and not data_cfg["columns"]["string_features"]:
        raise ConfigError(f"dataset '{name}': string_value_map needs columns.string_features")


def load_config(path):
    """Load the data config and every dataset config it lists.

    Returns the data config dict with an extra key `dataset_configs`: {dataset_name: dict}.
    """
    cfg = _read_yaml(path)
    validate_data_config(cfg)
    ds_dir = Path(cfg["paths"]["dataset_config_dir"])
    cfg["dataset_configs"] = {}
    for name in cfg["datasets"]:
        ds = _read_yaml(ds_dir / f"{name}.yaml")
        validate_dataset_config(ds, cfg, name)
        cfg["dataset_configs"][name] = ds
    return cfg


REQUIRED_MODEL_KEYS = ["data_config", "model_name", "encoding", "tasks", "seeds", "params", "tuning",
                       "leakage", "infer_time_per_rows", "outputs", "results_columns"]
REQUIRED_TASK_KEYS = ["target", "benign", "experiment", "cross_dataset", "eval_metric"]
REQUIRED_OUTPUT_KEYS = ["results_file", "run_details_dir", "tuning_file", "leakage_json", "leakage_md"]
REQUIRED_LEAKAGE_KEYS = ["eval_split", "flag_threshold", "seed", "params", "class_weight"]
CLASS_WEIGHTS = ["none", "balanced"]


def split_roles(data_cfg):
    """(train, val, test) split names: the data config `split` keys, in that order."""
    return tuple(data_cfg["split"])


def load_model_config(path):
    """Load a model config (e.g. configs/xgb.yaml) plus the data config it references (key `data`)."""
    cfg = _read_yaml(path)
    _require(cfg, REQUIRED_MODEL_KEYS, f"model config {path}")
    _require(cfg["outputs"], REQUIRED_OUTPUT_KEYS, f"model config {path} [outputs]")
    _require(cfg["leakage"], REQUIRED_LEAKAGE_KEYS, f"model config {path} [leakage]")
    for task, spec in cfg["tasks"].items():
        _require(spec, REQUIRED_TASK_KEYS, f"model config {path} [tasks.{task}]")
    cfg["data"] = load_config(cfg["data_config"])
    for task, spec in cfg["tasks"].items():
        for key in ["target", "benign"]:
            if spec[key] not in cfg["data"]["columns"]:
                raise ConfigError(f"tasks.{task}.{key} '{spec[key]}' is not a key of data config `columns`")
    if not cfg["seeds"]:
        raise ConfigError("seeds must not be empty")
    _require(cfg["tuning"]["grid"], ["class_weight"], f"model config {path} [tuning.grid]")
    weights = set(cfg["tuning"]["grid"]["class_weight"]) | {cfg["leakage"]["class_weight"]}
    if weights - set(CLASS_WEIGHTS):
        raise ConfigError(f"class_weight values must be in {CLASS_WEIGHTS}")
    train, val, test = split_roles(cfg["data"])
    if cfg["leakage"]["eval_split"] not in (train, val):
        raise ConfigError(f"leakage.eval_split must be '{train}' or '{val}', never the test split '{test}'")
    return cfg


ENCODER_OVERRIDABLE = ["format", "hash", "binning", "kv_text"]   # sections an encodings entry may override
REQUIRED_ENCODER_KEYS = ["data_config", "encodings", "format", "hash", "binning", "kv_text", "sentence",
                         "length_report", "outputs", "protocol_names", "l7_names"]


def _template_fields(sentences):
    return [f for s in sentences for _, f, _, _ in string.Formatter().parse(s) if f]


def load_encoder_config(path):
    """Load configs/encoders.yaml plus the data config it references (key `data`)."""
    cfg = _read_yaml(path)
    _require(cfg, REQUIRED_ENCODER_KEYS, f"encoder config {path}")
    _require(cfg["format"], ["separator", "token_joiner", "na_token", "float_decimals"], "encoder config [format]")
    _require(cfg["hash"], ["algorithm", "length"], "encoder config [hash]")
    _require(cfg["binning"], ["n_bins", "bin_prefix", "quantile_method", "state_dir", "categorical_columns"], "encoder config [binning]")
    _require(cfg["kv_text"], ["kv_separator", "use_aliases", "aliases"], "encoder config [kv_text]")
    _require(cfg["sentence"], ["derived", "template"], "encoder config [sentence]")
    cfg["data"] = load_config(cfg["data_config"])
    validate_encoder_config(cfg)
    return cfg


def validate_encoder_config(cfg):
    """Cross-checks against the data config's feature list (cfg must contain `data`)."""
    features = cfg["data"]["columns"]["features"]
    for name, spec in cfg["encodings"].items():
        if not isinstance(spec, dict):
            continue
        _require(spec, ["kind"], f"encoder config [encodings.{name}]")
        for section, values in spec.items():
            if section == "kind":
                continue
            if section not in ENCODER_OVERRIDABLE or set(values) - set(cfg[section]):
                raise ConfigError(f"encodings.{name}: may only override existing keys of {ENCODER_OVERRIDABLE}, "
                                  f"got {section}: {sorted(values)}")
    for key, cols in [("binning.categorical_columns", cfg["binning"]["categorical_columns"]),
                      ("kv_text.aliases", list(cfg["kv_text"]["aliases"]))]:
        unknown = sorted(set(cols) - set(features))
        if unknown:
            raise ConfigError(f"{key}: not features: {unknown}")

    not_binned = sorted(set(cfg["data"]["columns"]["string_features"]) - set(cfg["binning"]["categorical_columns"]))
    if not_binned:
        raise ConfigError(f"binning.categorical_columns must include every string feature (cannot bin text): {not_binned}")

    names = list(cfg["kv_text"]["aliases"].values())
    if len(set(names)) != len(names):
        raise ConfigError("kv_text.aliases must be unique")

    derived = cfg["sentence"]["derived"]
    for name, spec in derived.items():
        _require(spec, ["source", "names", "unknown"], f"encoder config [sentence.derived.{name}]")
        if spec["names"] not in cfg:
            raise ConfigError(f"sentence.derived.{name}.names '{spec['names']}' is not a top-level config key")
    used = [derived[f]["source"] if f in derived else f for f in _template_fields(cfg["sentence"]["template"])]
    unknown = sorted(set(used) - set(features))
    twice = sorted({f for f in used if used.count(f) > 1})
    missing = [f for f in features if f not in used]
    if unknown or twice or missing:
        raise ConfigError(f"E4 template must use every feature exactly once: unknown {unknown}, "
                          f"repeated {twice}, missing {missing}")


TEXT_MODEL_KEYS = ["data_config", "encoder_config", "model_name", "encodings", "tasks", "seeds", "tokenizer",
                   "model", "training", "tuning", "side_runs", "dataset_encodings", "infer_time_per_rows", "outputs",
                   "results_columns"]
TEXT_TRAINING_KEYS = ["device", "batch_size", "eval_batch_size", "grad_accum_steps", "epochs", "warmup_ratio",
                      "weight_decay", "grad_clip", "amp_dtype", "class_weight", "deterministic"]


def _load_text_model_config(path, extra_keys, training_keys, output_keys):
    """Checks shared by the text classifiers (small BERT, DistilBERT); adds `enc` and `data`."""
    cfg = _read_yaml(path)
    _require(cfg, TEXT_MODEL_KEYS + extra_keys, f"text model config {path}")
    _require(cfg["training"], TEXT_TRAINING_KEYS + training_keys, f"{path} [training]")
    _require(cfg["tuning"], ["lr_grid", "scope", "pilot"], f"{path} [tuning]")
    _require(cfg["tuning"]["pilot"], ["dataset", "task"], f"{path} [tuning.pilot]")
    if cfg["tuning"]["scope"] not in ("full", "pilot"):
        raise ConfigError("tuning.scope must be 'full' or 'pilot'")
    _require(cfg["outputs"], ["results_file", "run_details_dir", "tuning_file"] + output_keys, f"{path} [outputs]")
    for task, spec in cfg["tasks"].items():
        _require(spec, ["target", "benign", "experiment", "cross_dataset"], f"{path} [tasks.{task}]")
    cfg["enc"] = load_encoder_config(cfg["encoder_config"])
    cfg["data"] = cfg["enc"]["data"]
    if cfg["data_config"] != cfg["enc"]["data_config"]:
        raise ConfigError(f"{path}: data_config differs from the encoder config's data_config")
    unknown = sorted(set(cfg["encodings"]) - set(cfg["enc"]["encodings"]))
    if unknown:
        raise ConfigError(f"encodings not in the encoder config: {unknown}")
    for task, spec in cfg["tasks"].items():
        for key in ["target", "benign"]:
            if spec[key] not in cfg["data"]["columns"]:
                raise ConfigError(f"tasks.{task}.{key} '{spec[key]}' is not a key of data config `columns`")
    pilot = cfg["tuning"]["pilot"]
    if pilot["dataset"] not in cfg["data"]["datasets"] or pilot["task"] not in cfg["tasks"]:
        raise ConfigError(f"tuning.pilot {pilot} must name a configured dataset and task")
    if cfg["training"]["class_weight"] not in CLASS_WEIGHTS:
        raise ConfigError(f"training.class_weight must be in {CLASS_WEIGHTS}")
    if not cfg["seeds"] or not cfg["tuning"]["lr_grid"]:
        raise ConfigError("seeds and tuning.lr_grid must not be empty")
    for enc, spec in cfg["side_runs"].items():
        _require(spec, ["lr_from", "datasets", "tasks", "seeds"], f"{path} [side_runs.{enc}]")
        main = [e for e in cfg["encodings"] if e not in cfg["side_runs"]]
        if enc not in cfg["encodings"] or spec["lr_from"] not in main:
            raise ConfigError(f"side_runs.{enc}: must be a listed encoding with lr_from one of {main}")
        if set(spec["datasets"]) - set(cfg["data"]["datasets"]) or set(spec["tasks"]) - set(cfg["tasks"]):
            raise ConfigError(f"side_runs.{enc}: unknown datasets or tasks")
        if set(spec["seeds"]) - set(cfg["seeds"]):
            raise ConfigError(f"side_runs.{enc}: seeds must be a subset of seeds {cfg['seeds']}")
    for ds, encs in cfg["dataset_encodings"].items():
        if ds not in cfg["data"]["datasets"] or set(encs) - set(cfg["encodings"]):
            raise ConfigError(f"dataset_encodings.{ds}: must be a configured dataset with listed encodings")
    return cfg


def load_bert_config(path):
    """Load configs/bert_small.yaml plus the encoder config (key `enc`) and its data config (key `data`)."""
    cfg = _load_text_model_config(path, [], [], ["tokenizer_stats_json", "tokenizer_stats_md"])
    _require(cfg["tokenizer"], ["vocab_size", "min_frequency", "special_tokens", "coverage_percentile", "dir"],
             "bert config [tokenizer]")
    _require(cfg["tokenizer"]["special_tokens"], ["pad", "unk", "cls", "sep"], "bert config [tokenizer.special_tokens]")
    return cfg


def load_distilbert_config(path):
    """Load configs/distilbert.yaml plus the encoder config (key `enc`) and its data config (key `data`)."""
    cfg = _load_text_model_config(path, ["pretrained", "smoke"], ["gradient_checkpointing"],
                                  ["smoke_json", "smoke_md"])
    _require(cfg["tokenizer"], ["max_len_cap", "coverage_percentile"], "distilbert config [tokenizer]")
    _require(cfg["smoke"], ["encodings", "dataset", "task", "train_rows", "steps", "warmup_steps", "batch_sizes",
                            "eval_rows", "lr"], "distilbert config [smoke]")
    unknown = sorted(set(cfg["smoke"]["encodings"]) - set(cfg["encodings"]))
    if unknown:
        raise ConfigError(f"smoke.encodings not in encodings: {unknown}")
    return cfg


REQUIRED_PIPELINE_KEYS = ["default_target", "dims", "show_command", "groups", "stages"]
REQUIRED_STAGE_KEYS = ["name", "module", "config", "loader", "args", "dims", "units", "done", "force_flag",
                       "test_split", "depends_on", "est_minutes"]


def _check_ref(ref, where):
    module, _, func = (ref or "").partition(":")
    if not module or not func:
        raise ConfigError(f"{where}: '{ref}' must look like 'package.module:function'")


def load_pipeline_config(path):
    """Load configs/pipeline.yaml (stage list for src/run.py). Hooks are checked for form, imported on use."""
    cfg = _read_yaml(path)
    _require(cfg, REQUIRED_PIPELINE_KEYS, f"pipeline config {path}")
    seen = []
    for stage in cfg["stages"]:
        _require(stage, REQUIRED_STAGE_KEYS, f"pipeline config {path} [stage {stage.get('name')}]")
        name = stage["name"]
        if name in seen:
            raise ConfigError(f"stage '{name}' is listed twice")
        later = [d for d in stage["depends_on"] if d not in seen]
        if later:
            raise ConfigError(f"stage '{name}': depends_on {later} must be stages listed before it")
        unknown = [d for d in stage["dims"] if d not in cfg["dims"]]
        if unknown:
            raise ConfigError(f"stage '{name}': dims {unknown} not in {list(cfg['dims'])}")
        for key in ["loader", "done"] + (["units"] if stage["units"] else []):
            _check_ref(stage[key], f"stage '{name}' {key}")
        est = stage["est_minutes"]
        values = est.values() if isinstance(est, dict) else [est]
        if not all(isinstance(v, (int, float)) and v >= 0 for v in values):
            raise ConfigError(f"stage '{name}': est_minutes must be a number or a map of numbers")
        seen.append(name)
    for group, members in cfg["groups"].items():
        if group in seen:
            raise ConfigError(f"group '{group}' has the same name as a stage")
        unknown = [m for m in members if m not in seen]
        if unknown:
            raise ConfigError(f"group '{group}': unknown stages {unknown}")
    if cfg["default_target"] not in cfg["groups"]:
        raise ConfigError(f"default_target '{cfg['default_target']}' is not a group")
    return cfg


REQUIRED_ANALYSIS_KEYS = {
    "inputs": ["results_file", "run_details_dir", "required_columns"],
    "within": ["experiments", "task", "metrics"],
    "cross": ["experiment", "task", "datasets", "metrics"],
    "replication": ["experiment", "versions", "tasks", "metrics", "paper", "planned_from"],
    "side_runs": ["encodings", "main_label", "side_label"],
    "format": ["decimals", "std_ddof", "one_seed_marker", "tex_one_seed_marker", "pending", "not_run", "max_seeds_note",
               "lower_is_better", "metric_labels"],
    "errors": ["confusion_models", "top_confusions", "hard_class_f1"],
    "hypotheses": ["tie_margin", "well_margin", "baseline_model", "text_models", "pretrained_model",
                   "scratch_model", "ppfle_encoding", "alternative_encodings", "readable_encodings",
                   "hashed_encodings"],
    "figures": ["dir", "formats", "dpi", "font_size", "model_colors", "model_markers", "sequential_ramp",
                "text_primary", "text_secondary", "text_on_dark", "dark_cell_threshold", "grid", "surface",
                "heatmap_range", "marker_size", "min_axis_span", "annot_decimals", "annot_font_size", "confusion_min_share", "style",
                "sizes"],
    "timing": ["seeds", "cluster"],
    "outputs": ["exp2_table", "side_runs_table", "exp1_table", "exp3_table", "error_analysis", "summary", "tex_dir"],
    "examples": ["source", "tex", "max_chars", "monospace", "text_width"],
    "numbers": ["tex", "md", "macro", "float_decimals", "percent_decimals", "time_decimals", "params_decimals",
                "max_string_length", "sources", "report_files", "all_numbers"],
}


def load_analysis_config(path):
    """Load configs/analysis.yaml (tables, figures and H1-H4 summary from results.csv; src/eval/analysis.py)."""
    cfg = _read_yaml(path)
    _require(cfg, list(REQUIRED_ANALYSIS_KEYS) + ["datasets", "methods", "models", "encodings"],
             f"analysis config {path}")
    for section, keys in REQUIRED_ANALYSIS_KEYS.items():
        _require(cfg[section], keys, f"analysis config {path} [{section}]")
    for m in cfg["methods"]:
        _require(m, ["model", "encoding", "label"], f"analysis config {path} [methods]")
        if m["model"] not in cfg["models"] or m["encoding"] not in cfg["encodings"]:
            raise ConfigError(f"method {m}: model/encoding missing from `models`/`encodings`")
    hyp, fig = cfg["hypotheses"], cfg["figures"]
    models = [hyp["baseline_model"], hyp["pretrained_model"], hyp["scratch_model"], *hyp["text_models"],
              *cfg["errors"]["confusion_models"], *fig["model_colors"], *fig["model_markers"]]
    unknown = sorted(set(models) - set(cfg["models"]))
    if unknown:
        raise ConfigError(f"analysis config {path}: unknown models {unknown}")
    encodings = [hyp["ppfle_encoding"], *hyp["alternative_encodings"], *hyp["readable_encodings"],
                 *hyp["hashed_encodings"], *cfg["side_runs"]["encodings"].values()]
    unknown = sorted(set(encodings) - set(cfg["encodings"]))
    if unknown:
        raise ConfigError(f"analysis config {path}: unknown encodings {unknown}")
    unknown = sorted(set(cfg["cross"]["datasets"]) - set(cfg["datasets"]))
    if unknown:
        raise ConfigError(f"analysis config {path}: cross.datasets not in `datasets`: {unknown}")
    metrics = cfg["within"]["metrics"] + cfg["cross"]["metrics"] + cfg["replication"]["metrics"]
    unknown = sorted(set(metrics) - set(cfg["inputs"]["required_columns"]))
    if unknown:
        raise ConfigError(f"analysis config {path}: metrics not in inputs.required_columns: {unknown}")
    if not fig["formats"]:
        raise ConfigError(f"analysis config {path}: figures.formats must not be empty")
    return cfg


REQUIRED_DEMO_KEYS = ["xgb_config", "bert_config", "privacy_config", "analysis_config", "dataset", "task", "seed",
                      "xgb", "encodings", "model_dir", "meta_file", "xgb_model_file", "app", "leak", "mystery", "story"]
REQUIRED_DEMO_APP_KEYS = ["title", "device", "xgb_device", "show_encodings", "attack_encoding", "top_k", "decimals",
                          "percent_decimals", "random_seed", "encode_step_s", "piece_colors", "column_info", "figures",
                          "summary_file"]
REQUIRED_DEMO_STORY_KEYS = ["steps", "card", "byte_units", "port_names", "classes", "encodings", "highlight",
                            "classifiers", "what_if", "what_if_port", "port_guess", "missing_number", "findings"]


def load_demo_config(path):
    """Load configs/demo.yaml (live demo, src/demo/) plus the configs it points to: `xgb_cfg`, `bert_cfg`,
    `privacy_cfg` (each loaded with its own loader), the Edge-IIoTset data config under `leak_data` and the
    analysis config's display names under `display`."""
    cfg = _read_yaml(path)
    _require(cfg, REQUIRED_DEMO_KEYS, f"demo config {path}")
    _require(cfg["app"], REQUIRED_DEMO_APP_KEYS, f"demo config {path} [app]")
    _require(cfg["leak"], ["data_config", "raw_dataset", "fixed_dataset", "split"], f"demo config {path} [leak]")
    _require(cfg["story"], REQUIRED_DEMO_STORY_KEYS, f"demo config {path} [story]")
    cfg["xgb_cfg"] = load_model_config(cfg["xgb_config"])
    cfg["bert_cfg"] = load_bert_config(cfg["bert_config"])
    cfg["privacy_cfg"] = load_privacy_config(cfg["privacy_config"])
    cfg["leak_data"] = load_config(cfg["leak"]["data_config"])
    style = load_analysis_config(cfg["analysis_config"])
    cfg["display"] = {k: style[k] for k in ("datasets", "models", "encodings")}
    cfg["numbers_files"] = [style["numbers"]["tex"], cfg["privacy_cfg"]["numbers"]["tex"]]   # story.findings keys
    bert = cfg["bert_cfg"]
    if cfg["dataset"] not in bert["data"]["datasets"] or cfg["task"] not in bert["tasks"]:
        raise ConfigError(f"demo config {path}: dataset/task must be configured in {cfg['bert_config']}")
    unknown = sorted(set(cfg["encodings"]) - set(bert["encodings"]))
    if unknown:
        raise ConfigError(f"demo config {path}: encodings without small BERT tokenizers: {unknown}")
    shown = set(cfg["app"]["show_encodings"]) | {cfg["app"]["attack_encoding"]}
    unknown = sorted(shown - set(bert["enc"]["encodings"]))
    if unknown:
        raise ConfigError(f"demo config {path}: app encodings not in the encoder config: {unknown}")
    features = bert["data"]["columns"]["features"]
    if set(cfg["app"]["column_info"]) != set(features):
        raise ConfigError(f"demo config {path}: app.column_info needs one entry per feature, missing "
                          f"{sorted(set(features) - set(cfg['app']['column_info']))}, unknown "
                          f"{sorted(set(cfg['app']['column_info']) - set(features))}")
    story = cfg["story"]
    used = set(story["card"].values()) | set(story["highlight"]) | set(story["what_if"]) | {
        story["what_if_port"], story["port_guess"]["column"]}
    if used - set(features):
        raise ConfigError(f"demo config {path}: story features not in the data config: {sorted(used - set(features))}")
    if set(cfg["app"]["show_encodings"]) - set(story["encodings"]):
        raise ConfigError(f"demo config {path}: story.encodings needs an entry per app.show_encodings encoding")
    models = {(cfg["xgb_cfg"]["model_name"], cfg["xgb_cfg"]["encoding"])} if cfg["xgb"] else set()
    models |= {(bert["model_name"], e) for e in cfg["encodings"]}
    if {(d["model"], d["encoding"]) for d in story["classifiers"]} != models:
        raise ConfigError(f"demo config {path}: story.classifiers needs exactly one entry per demo model {sorted(models)}")
    if story["port_guess"]["column"] not in cfg["privacy_cfg"]["low_cardinality"]:
        raise ConfigError(f"demo config {path}: story.port_guess.column needs a low_cardinality domain in "
                          f"{cfg['privacy_config']}")
    if cfg["app"]["attack_encoding"] not in cfg["privacy_cfg"]["encodings"]:
        raise ConfigError(f"demo config {path}: app.attack_encoding must be attacked in {cfg['privacy_config']}")
    leak = cfg["leak"]
    if {leak["raw_dataset"], leak["fixed_dataset"]} - set(cfg["leak_data"]["datasets"]):
        raise ConfigError(f"demo config {path}: leak datasets must be in {leak['data_config']}")
    return cfg


REQUIRED_PRIVACY_KEYS =["encoder_config", "style_from", "target_split", "public_split", "public_datasets",
                         "dataset_labels", "encodings", "levels", "low_cardinality", "level_style", "bruteforce_range",
                         "outputs"]
PRIVACY_DOMAIN_KINDS = ["range", "dotted_range", "hex_range", "values"]   # see configs/privacy.yaml low_cardinality
PRIVACY_LEVELS = ["enumerate", "bruteforce", "public"]   # the attack steps src/eval/privacy.py implements, in order


def load_privacy_config(path):
    """Load configs/privacy.yaml (Exp 4 dictionary attack) plus the encoder config (hash scheme, bins, data config,
    under `encoder` and `data`) and the figure style of the analysis config (under `figures`)."""
    cfg = _read_yaml(path)
    _require(cfg, REQUIRED_PRIVACY_KEYS, f"privacy config {path}")
    _require(cfg["outputs"], ["json", "table", "tex", "figure", "label"], f"privacy config {path} [outputs]")
    _require(cfg, ["numbers"], f"privacy config {path}")
    _require(cfg["numbers"], ["tex", "md", "macro", "prefix", "float_decimals", "percent_decimals",
                              "max_string_length"],
             f"privacy config {path} [numbers]")
    cfg["encoder"] = load_encoder_config(cfg["encoder_config"])
    cfg["data"] = cfg["encoder"]["data"]
    style = load_analysis_config(cfg["style_from"])
    cfg["figures"], cfg["display"] = style["figures"], {k: style[k] for k in ("datasets", "encodings")}
    names = [lv["name"] for lv in cfg["levels"]]
    if not names or names != [n for n in PRIVACY_LEVELS if n in names]:
        raise ConfigError(f"privacy config {path}: levels must be some of {PRIVACY_LEVELS}, in this order")
    if not (len(cfg["level_style"]["colors"]) == len(cfg["level_style"]["markers"]) == len(cfg["levels"])):
        raise ConfigError(f"privacy config {path}: level_style needs one colour and marker per level")
    unknown = sorted(set(cfg["encodings"]) - set(cfg["encoder"]["encodings"]))
    if unknown:
        raise ConfigError(f"privacy config {path}: unknown encodings {unknown}")
    features = cfg["data"]["columns"]["features"]
    for col, spec in cfg["low_cardinality"].items():
        if col not in features:
            raise ConfigError(f"privacy config {path}: low_cardinality column {col} is not a feature")
        if not spec or set(spec) - set(PRIVACY_DOMAIN_KINDS):
            raise ConfigError(f"privacy config {path}: {col} needs one or more of {PRIVACY_DOMAIN_KINDS}")
    unknown = sorted(set(cfg["public_datasets"]) - set(cfg["data"]["datasets"]))
    if unknown:
        raise ConfigError(f"privacy config {path}: public_datasets not in the data config: {unknown}")
    split_names = split_roles(cfg["data"])
    if cfg["target_split"] not in split_names or cfg["public_split"] not in split_names:
        raise ConfigError(f"privacy config {path}: target/public split must be one of {split_names}")
    return cfg

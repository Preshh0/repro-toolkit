#!/usr/bin/env python3
"""repro: a small toolkit for the Reproducible AI Research checklist (stdlib only).

CLI:
  repro init NAME          scaffold a project (docs templates, config, toy train.py)
  repro decide "text"      append a timestamped line to DECISIONS.md
  repro check [--core]     audit the project against the checklist
  repro tick 4.2 [...]     manually confirm items that can't be auto-checked
  repro untick 4.2 [...]
  repro report             mean +/- std of metrics across seeds
  repro mof [--target II]  classify as MOF Class III/II/I; list what is missing and how to fix it, per config

Library:
  from repro import Run
  with Run("config.json") as run:
      ...train...
      run.log_metric("val_accuracy", 0.91)
      run.evaluate_test(lambda: model.score(X_test, y_test))   # allowed ONCE
"""
import argparse, hashlib, json, math, os, platform, random, re, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

STATE = ".repro/checklist.json"


# ----------------------------------------------------------------- helpers
def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sh(cmd, cwd="."):
    try:
        return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=120).stdout.strip()
    except Exception:
        return ""


def sha256_path(path):
    """Hash a file, or a directory (names + contents). Skips hidden files and MANIFEST.json."""
    p = Path(path)
    if not p.exists():
        return None
    h = hashlib.sha256()
    files = [p] if p.is_file() else sorted(
        f for f in p.rglob("*")
        if f.is_file() and f.name != "MANIFEST.json" and not any(x.startswith(".") for x in f.parts))
    for f in files:
        h.update(str(f.relative_to(p) if p.is_dir() else f.name).encode())
        with open(f, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


def config_hash(cfg):
    """Hash of the config WITHOUT the seed, so runs differing only by seed group together."""
    c = {k: v for k, v in cfg.items() if k != "seed"}
    return hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest()[:12]


def set_seeds(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ----------------------------------------------------------------- Run (section 4 + 5)
class TestSetReused(RuntimeError):
    pass


class Run:
    """Context manager that writes runs/<id>/record.json: seed, commit, data hash,
    hyperparameters, environment, metrics, status. Failed runs are recorded too."""

    def __init__(self, config="config.json", root=".", seed=None):
        self.root = Path(root)
        self.cfg = json.loads((self.root / config).read_text())
        if seed is not None:
            self.cfg["seed"] = seed
        self.metrics, self.notes, self.test_result = {}, [], None

    def __enter__(self):
        c = self.cfg
        set_seeds(c["seed"])
        self.t0 = time.time()
        self.id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3] + f"-s{c['seed']}"
        self.dir = self.root / "runs" / self.id
        self.dir.mkdir(parents=True, exist_ok=True)
        data_path = self.root / c.get("data", {}).get("path", "data")
        self.prov = {
            "commit": sh(["git", "rev-parse", "HEAD"], self.root) or None,
            "git_dirty": bool(sh(["git", "status", "--porcelain"], self.root)),
            "data_version": c.get("data", {}).get("version"),
            "data_sha256": sha256_path(data_path),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        }
        (self.dir / "environment.txt").write_text(sh([sys.executable, "-m", "pip", "freeze"]))
        (self.dir / "config.json").write_text(json.dumps(c, indent=2, sort_keys=True))
        return self

    def log_metric(self, name, value):
        self.metrics[name] = value

    def log_note(self, text):
        self.notes.append(text)

    def evaluate_test(self, fn, force=False):
        """Run fn() on the test set. Refuses a second evaluation unless force=True (which is logged)."""
        ledger_path = self.root / "runs" / "test_ledger.json"
        ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else []
        if ledger and not force:
            raise TestSetReused(
                f"Test set already evaluated in run {ledger[0]['run']} at {ledger[0]['time']}. "
                "Pass force=True only if you will report that it was re-used.")
        self.test_result = fn()
        ledger.append({"run": self.id, "time": now(), "forced": bool(ledger)})
        ledger_path.write_text(json.dumps(ledger, indent=2))
        self.metrics["test"] = self.test_result
        return self.test_result

    def __exit__(self, etype, exc, tb):
        rec = {
            "id": self.id, "status": "failed" if etype else "ok",
            "error": f"{etype.__name__}: {exc}" if etype else None,
            "started": datetime.fromtimestamp(self.t0, timezone.utc).isoformat(timespec="seconds"),
            "seconds": round(time.time() - self.t0, 2),
            "seed": self.cfg["seed"], "config_hash": config_hash(self.cfg),
            "hyperparameters": self.cfg.get("hyperparameters", {}),
            "metric_definition": self.cfg.get("metric"),
            "metrics": self.metrics, "notes": self.notes, **self.prov,
        }
        (self.dir / "record.json").write_text(json.dumps(rec, indent=2))
        print(f"[repro] {rec['status']} -> {self.dir / 'record.json'}")
        return False  # never swallow exceptions


def load_runs(root="."):
    out = []
    for f in sorted(Path(root, "runs").glob("*/record.json")):
        try:
            out.append(json.loads(f.read_text()))
        except Exception:
            pass
    return out


# ----------------------------------------------------------------- doc-section helpers
def section(path, heading):
    """Body text under '## heading' (HTML comments stripped), or None if absent."""
    p = Path(path)
    if not p.exists():
        return None
    buf, on, found = [], False, False
    for line in p.read_text().splitlines():
        if line.startswith("#"):
            if on:
                break
            on = found = line.lstrip("# ").strip().lower() == heading.lower()
            continue
        if on:
            buf.append(line)
    if not found:
        return None
    return re.sub(r"<!--.*?-->", "", "\n".join(buf), flags=re.S).strip()


def filled(file, *headings):
    def check(root):
        missing = []
        for h in headings:
            body = section(Path(root, file), h)
            if not body or "TODO" in body:
                missing.append(h)
        return (not missing, f"fill in {file} -> " + ", ".join(f"'{m}'" for m in missing) if missing else "")
    return check


def exists(*names, hint=None):
    return lambda root: (any(Path(root, n).exists() for n in names), hint or f"add {' or '.join(names)}")


def cfg_of(root):
    try:
        return json.loads(Path(root, "config.json").read_text())
    except Exception:
        return {}


def latest_ok(root):
    ok = [r for r in load_runs(root) if r["status"] == "ok"]
    return ok[-1] if ok else None


def c_seed(root):
    return ("seed" in cfg_of(root), "add 'seed' to config.json")


def c_hyper(root):
    return (bool(cfg_of(root).get("hyperparameters")), "put hyperparameters under config.json -> 'hyperparameters'")


def c_commit(root):
    r = latest_ok(root)
    return (bool(r and r.get("commit")), "no run with a git commit yet: git init, commit, then run")


def c_clean(root):
    r = latest_ok(root)
    return (bool(r and r.get("commit") and not r.get("git_dirty")), "latest run used uncommitted code: commit, re-run")


def c_snapshot(root):
    r = latest_ok(root)
    return (bool(r and r.get("data_sha256") and r.get("data_version")), "set config data.version and keep data under data.path")


def c_env(root):
    r = latest_ok(root)
    return (bool(r and (Path(root, "runs", r["id"], "environment.txt")).stat().st_size > 0), "no environment.txt captured")


def c_testonce(root):
    p = Path(root, "runs", "test_ledger.json")
    n = len(json.loads(p.read_text())) if p.exists() else 0
    return (n <= 1, f"test set evaluated {n} times: report this honestly and consider a fresh split")


def c_seeds(root):
    groups = {}
    for r in load_runs(root):
        if r["status"] == "ok":
            groups.setdefault(r["config_hash"], set()).add(r["seed"])
    return (any(len(s) >= 3 for s in groups.values()), "run the same config with >=3 seeds: repro report shows variance")


def c_decisions(root):
    p = Path(root, "DECISIONS.md")
    return (p.exists() and len(re.findall(r"^- \[", p.read_text(), flags=re.M)) >= 1, "log a decision: repro decide \"...\"")


def c_entry(root):
    ep = cfg_of(root).get("entry_point")
    return (bool(ep and Path(root, ep).exists()), "set config 'entry_point' to an existing script")


# (id, text, core?, auto-check or None)
CHECKLIST = [
    ("1", "Research Question", [
        ("1.1", "The question is specific and answerable — not a broad topic", True, filled("RESEARCH_QUESTION.md", "Question")),
        ("1.2", "Success metric defined before any training begins", False, filled("RESEARCH_QUESTION.md", "Success metric")),
        ("1.3", "Scope is bounded: time period, geography, or population stated", False, filled("RESEARCH_QUESTION.md", "Scope")),
        ("1.4", "A null, negative result, or error would also be documented and shared", True, filled("RESEARCH_QUESTION.md", "Null or negative result plan")),
    ]),
    ("2", "Data Provenance", [
        ("2.1", "Data source and collection method are documented", True, filled("DATA_CARD.md", "Source and collection")),
        ("2.2", "Date range and dataset version are recorded", True, filled("DATA_CARD.md", "Date range and version")),
        ("2.3", "Inclusion and exclusion criteria are written down", True, filled("DATA_CARD.md", "Inclusion and exclusion criteria")),
        ("2.4", "Licence or terms of use are confirmed before use", True, filled("DATA_CARD.md", "Licence")),
        ("2.5", "Preprocessing steps are recorded, not just described", True, filled("DATA_CARD.md", "Preprocessing")),
    ]),
    ("3", "Documentation", [
        ("3.1", "README covers: what, why, and how to run", True, filled("README.md", "What", "Why", "How to run")),
        ("3.2", "Column names, units, and label definitions are explained", True, filled("DATA_CARD.md", "Columns and labels")),
        ("3.3", "Known limitations are stated honestly", True, filled("LIMITATIONS.md", "Limitation statement")),
        ("3.4", "Bias checks have been performed and documented", True, filled("DATA_CARD.md", "Bias checks")),
        ("3.5", "Written for someone with no prior context (could a new member run it in one day?)", True, None),
    ]),
    ("4", "Experiment Record", [
        ("4.1", "Seed fixed and logged in the config file", True, c_seed),
        ("4.2", "Commit recorded per run (and tree was clean)", True, c_clean),
        ("4.3", "Data snapshot tagged or versioned, not just a file path", True, c_snapshot),
        ("4.4", "Hyperparameters saved in a config file, not hardcoded", True, c_hyper),
        ("4.5", "Environment exported (pip freeze / conda env export)", True, c_env),
        ("4.6", "Metric definition is unambiguous and lives in the code", True, exists("metrics.py", hint="put the metric in metrics.py (and name it in config 'metric')")),
    ]),
    ("5", "Honest Reporting", [
        ("5.1", "Tuning procedure described fully: grid size, folds, timing", True, filled("REPORT.md", "Tuning procedure")),
        ("5.2", "Reported metric matches the evaluation protocol exactly", True, filled("REPORT.md", "Evaluation protocol")),
        ("5.3", "Test set evaluated only once, after all decisions were made", True, c_testonce),
        ("5.4", "Variance across seeds reported where feasible", True, c_seeds),
        ("5.5", "Negative results and failure cases are noted", True, filled("REPORT.md", "Negative results and failure cases")),
    ]),
    ("6", "Privacy and Sharing", [
        ("6.1", "Re-identification risk assessed for all data fields", True, filled("DATA_CARD.md", "Re-identification risk")),
        ("6.2", "Consent or licence in place before sharing", True, filled("DATA_CARD.md", "Sharing plan")),
        ("6.3", "Sharing strategy consistent with the assessed risk level", True, None),
        ("6.4", "Code is released under an explicit open licence", True, exists("LICENSE", "LICENSE.md", "LICENSE.txt", hint="add a LICENSE file (e.g. MIT, Apache-2.0)")),
        ("6.5", "Sensitive fields redacted, aggregated, or replaced with synthetic data", True, None),
    ]),
    ("7", "Release Package", [
        ("7.1", "Source code is in a version-controlled repository", True, exists(".git", hint="git init && git add -A && git commit")),
        ("7.2", "Data pointer included: URL, DOI, or access instructions", True, filled("DATA_CARD.md", "Data pointer")),
        ("7.3", "A single entry-point script reproduces the main result (<= 3 commands)", True, c_entry),
        ("7.4", "Environment specification file is included", True, exists("requirements.txt", "environment.yml", "pyproject.toml", hint="add requirements.txt (pip freeze > requirements.txt)")),
        ("7.5", "Random seed is hard-coded or in the config", True, c_seed),
        ("7.6", "One honest limitation statement written for reviewers", True, filled("LIMITATIONS.md", "Limitation statement")),
    ]),
    ("8", "Process and Culture", [
        ("8.1", "Decisions are logged as they are made, not reconstructed afterwards", True, c_decisions),
        ("8.2", "A collaborator can reproduce the result without asking you", True, None),
        ("8.3", "Reproducibility is treated as part of done, not a final clean-up", True, None),
        ("8.4", "Trade-offs and constraints are documented, not hidden", True, None),
        ("8.5", "This checklist has been reviewed by a team member before submission", True, None),
        ("8.6", "The checklist has been adapted to the specific needs of this project", True, None),
    ]),
]


# ----------------------------------------------------------------- Model Openness Framework (MOF)
WEIGHT_EXT = (".pt", ".pth", ".safetensors", ".ckpt", ".bin", ".h5", ".onnx", ".gguf", ".pkl", ".joblib")
INTERMEDIATE = re.compile(r"(step|epoch|iter|ckpt-?\d|checkpoint-?\d|intermediate)", re.I)
SKIP = {".git", "runs", ".venv", "venv", "node_modules", "site-packages", "__pycache__", ".repro"}
URLISH = re.compile(r"https?://|doi\.org|10\.\d{4,}|arxiv", re.I)
DATA_HEADINGS = ("Source and collection", "Date range and version", "Inclusion and exclusion criteria", "Licence",
                 "Preprocessing", "Columns and labels", "Bias checks", "Re-identification risk", "Sharing plan", "Data pointer")


def has(root, *globs):
    for g in globs:
        for p in Path(root).glob(g):
            if p.is_file() and not SKIP & set(p.relative_to(root).parts):
                return True
    return False


def has_dir(root, *names):
    return any(Path(root, n).is_dir() and has(Path(root, n), "**/*") and
               any(f.is_file() and f.name != ".gitkeep" for f in Path(root, n).rglob("*")) for n in names)


def weights(root):
    return [p for p in Path(root).rglob("*") if p.is_file() and p.suffix.lower() in WEIGHT_EXT
            and not SKIP & set(p.relative_to(root).parts)]


def pointer(root, file, heading, url=False):
    body = section(Path(root, file), heading)
    return bool(body) and "TODO" not in body and (not url or bool(URLISH.search(body)))


def nonempty_lines(root, *names):
    for n in names:
        p = Path(root, n)
        if p.is_file() and any(l.strip() and not l.strip().startswith("#") for l in p.read_text().splitlines()):
            return True
    return False


def m_card(root):
    p = Path(root, "MODEL_CARD.md")
    return p.exists() and "TODO" not in p.read_text()


def m_train(root):
    cfg = cfg_of(root)
    split = has(root, "**/split*.py") or "splits" in cfg or "splits" in cfg.get("data", {})
    return has(root, "**/train*.py", "training/**/*.py") and bool(split)


# (id, component, optional?, detector(root)->bool, how to improve)
MOF = [
    ("III", "Open Model", [
        ("III.1", "Model Architecture", False,
         lambda r: pointer(r, "MODEL_CARD.md", "Architecture") or has(r, "**/model*.py", "**/architecture*"),
         "Describe the architecture under 'Architecture' in MODEL_CARD.md, or include the model definition (model.py) or config."),
        ("III.2", "Model Parameters (final checkpoint)", False,
         lambda r: any(not INTERMEDIATE.search(w.name) for w in weights(r)) or pointer(r, "MODEL_CARD.md", "Model parameters pointer"),
         "Put the final weights in checkpoints/ (e.g. final.safetensors), or add a hub/DOI link under 'Model parameters pointer' in MODEL_CARD.md."),
        ("III.3", "Technical Report or Research Paper", False,
         lambda r: has(r, "paper/**/*", "docs/**/*.pdf", "**/report*.pdf", "**/technical*report*")
         or pointer(r, "MODEL_CARD.md", "Research paper", url=True)
         or all(pointer(r, "REPORT.md", h) for h in ("Tuning procedure", "Evaluation protocol", "Negative results and failure cases")),
         "Write up the method and results (paper/ folder, a PDF, or a complete REPORT.md), or link it under 'Research paper' in MODEL_CARD.md."),
        ("III.4", "Evaluation Results", False,
         lambda r: has_dir(r, "results") or pointer(r, "MODEL_CARD.md", "Evaluation results"),
         "Save result tables/logs in results/, or fill 'Evaluation results' in MODEL_CARD.md (metric from metrics.py, mean ± std over seeds)."),
        ("III.5", "Model Card", False, m_card,
         "Complete MODEL_CARD.md: every TODO replaced with real content."),
        ("III.6", "Data Card", False,
         lambda r: all(pointer(r, "DATA_CARD.md", h) for h in DATA_HEADINGS),
         "Complete DATA_CARD.md: all ten sections filled, none left as TODO."),
        ("III.7", "Sample Model Outputs", True,
         lambda r: has_dir(r, "samples", "sample_outputs", "examples"),
         "Optional: add a few representative (and failure-case) outputs in samples/."),
    ]),
    ("II", "Open Tooling", [
        ("II.2", "Training, Validation and Testing Code", False, m_train,
         "Include the training script (train.py) AND how data is split (split.py, or a 'splits' entry in config.json)."),
        ("II.3", "Inference Code", False,
         lambda r: has(r, "**/infer*.py", "**/predict*.py", "**/serve*.py", "**/demo*.py", "inference/**/*.py"),
         "Add predict.py / infer.py showing how to load the final checkpoint and run it on new input."),
        ("II.4", "Evaluation Code", False,
         lambda r: has(r, "**/eval*.py", "**/benchmark*.py", "evaluation/**/*.py"),
         "Add evaluate.py that loads the checkpoint and reproduces the reported results (metrics.py alone is not enough)."),
        ("II.5", "Evaluation Data", False,
         lambda r: has_dir(r, "eval_data", "test_data", "evaluation_data") or has(r, "data/test*", "data/eval*", "data/**/test*", "data/**/eval*")
         or pointer(r, "DATA_CARD.md", "Evaluation data"),
         "Ship the evaluation/test split (eval_data/ or data/test*), or link it under 'Evaluation data' in DATA_CARD.md."),
        ("II.6", "Supporting Libraries & Tools", False,
         lambda r: nonempty_lines(r, "requirements.txt", "environment.yml", "Pipfile", "pyproject.toml") or has(r, "Dockerfile"),
         "Pin dependencies: pip freeze > requirements.txt (or environment.yml / Dockerfile)."),
    ]),
    ("I", "Open Science", [
        ("I.2", "Research Paper", False,
         lambda r: has(r, "paper/**/*", "**/paper*.pdf", "**/paper*.tex") or pointer(r, "MODEL_CARD.md", "Research paper", url=True),
         "Provide the full paper: paper/ folder (source or PDF), or an arXiv/DOI link under 'Research paper' in MODEL_CARD.md."),
        ("I.3", "Datasets", False,
         lambda r: has_dir(r, "data", "datasets") or pointer(r, "DATA_CARD.md", "Data pointer", url=True),
         "Release the training/validation/test datasets: files in data/, or a URL/DOI under 'Data pointer' in DATA_CARD.md."),
        ("I.4", "Data Preprocessing Code", False,
         lambda r: has(r, "**/preprocess*.py", "**/prepare*.py", "**/data_prep*.py", "preprocessing/**/*.py", "**/clean*.py"),
         "Add the script that turns raw data into the training data (preprocess.py) and link it from DATA_CARD.md → 'Preprocessing'."),
        ("I.5", "Model Parameters (intermediate checkpoints)", False,
         lambda r: sum(bool(INTERMEDIATE.search(w.name)) for w in weights(r)) >= 2
         or pointer(r, "MODEL_CARD.md", "Intermediate checkpoints pointer"),
         "Save checkpoints during training (e.g. step-1000.pt, step-2000.pt) in checkpoints/, or link them under 'Intermediate checkpoints pointer'."),
        ("I.6", "Model Metadata", True,
         lambda r: has(r, "metadata.*", "model_metadata.*", "**/croissant*.json"),
         "Optional: add metadata.json (or Croissant) describing model, data and training; runs/*/record.json is a good starting point."),
    ]),
]
MOF_IDS = [c[0] for _, _, comps in MOF for c in comps]
RANK = {"III": 1, "II": 2, "I": 3}
LABEL = {"III": "Class III · Open Model", "II": "Class II · Open Tooling", "I": "Class I · Open Science"}


def mof_eval(root, manual):
    out = []
    for cid, cname, comps in MOF:
        rows = []
        for iid, name, opt, fn, fix in comps:
            try:
                auto = bool(fn(root))
            except Exception:
                auto = False
            rows.append((iid, name, opt, "auto" if auto else ("manual" if iid in manual else None), fix))
        out.append((cid, rows))
    achieved, chain = None, True
    for cid, rows in out:  # III -> II -> I; each class requires the one below it
        chain = chain and all(r[3] for r in rows if not r[2])
        if chain:
            achieved = cid
    return out, achieved


def cmd_mof(a):
    out, achieved = mof_eval(a.root, ticked(a.root))
    print(f"\nMOF result: {LABEL[achieved] if achieved else 'not yet Class III (Open Model)'}")
    below_ok = True
    first_gap = None
    for cid, rows in out:
        req = [r for r in rows if not r[2]]
        missing = [r for r in req if not r[3]]
        note = "" if below_ok else "  (needs the class below first)"
        print(f"\n{LABEL[cid]}  {len(req) - len(missing)}/{len(req)} required{note}")
        if cid != "III":
            print(f"  \u2192 includes all components of the class below")
        for iid, name, opt, via, fix in rows:
            if via:
                print(f"  \u2714 {iid}  {name}  [{via}]")
            elif opt:
                print(f"  \u25cb {iid}  {name} (optional)\n        -> {fix}")
            else:
                print(f"  \u2718 {iid}  {name}\n        -> {fix}")
        if missing and first_gap is None:
            first_gap = (cid, missing)
        below_ok = below_ok and not missing
    left = sum(1 for _, rows in out for r in rows if not r[2] and not r[3])
    if achieved == "I":
        print("\nBest class reached. Re-run after changes; optional items above would strengthen it.")
    else:
        cid, miss = first_gap
        print(f"\nNext step: {len(miss)} required item(s) to reach {LABEL[cid]}: " + ", ".join(m[0] for m in miss))
        print(f"Total required items left for Class I (Open Science): {left}")
    lic = has(a.root, "LICENSE", "LICENSE.md", "LICENSE.txt")
    print("\nNote: MOF also expects each component under an open licence (code, data, model, docs). "
          f"Code LICENSE file: {'found' if lic else 'MISSING - add one'}. Use `repro tick <id>` for items the checker can't see (e.g. hosted elsewhere).")
    return 0 if achieved and RANK[achieved] >= RANK[a.target] else 1


# ----------------------------------------------------------------- commands
def ticked(root):
    p = Path(root, STATE)
    return set(json.loads(p.read_text())) if p.exists() else set()


def cmd_check(a):
    root, manual, fails, total = a.root, ticked(a.root), 0, 0
    for sid, title, items in CHECKLIST:
        rows = []
        for iid, text, core, fn in items:
            if a.core and not core:
                continue
            total += 1
            ok, hint = fn(root) if fn else (False, "manual: confirm, then `repro tick %s`" % iid)
            if ok:
                rows.append(f"  \u2714 {iid}  {text}  [auto]")
            elif iid in manual:
                rows.append(f"  \u2714 {iid}  {text}  [manual]")
            else:
                fails += 1
                rows.append(f"  \u2718 {iid}  {text}\n        -> {hint}")
        if rows:
            print(f"\n{sid} \u00b7 {title}")
            print("\n".join(rows))
    print(f"\n{total - fails}/{total} items satisfied.")
    return 1 if fails else 0


def cmd_tick(a, on=True):
    valid = {i[0] for _, _, items in CHECKLIST for i in items} | set(MOF_IDS)
    s = ticked(a.root)
    for i in a.ids:
        if i not in valid:
            sys.exit(f"unknown item {i}")
        (s.add if on else s.discard)(i)
    p = Path(a.root, STATE)
    p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps(sorted(s)))
    print(("ticked " if on else "unticked ") + " ".join(a.ids))


def cmd_decide(a):
    p = Path(a.root, "DECISIONS.md")
    if not p.exists():
        p.write_text("# Decision log\n\nAppend as you decide, not afterwards.\n\n")
    with open(p, "a") as f:
        f.write(f"- [{now()}] {a.text}\n")
    print("logged.")


def cmd_report(a):
    groups = {}
    for r in load_runs(a.root):
        groups.setdefault(r["config_hash"], []).append(r)
    if not groups:
        return print("no runs yet")
    for ch, rs in groups.items():
        ok = [r for r in rs if r["status"] == "ok"]
        print(f"\nconfig {ch}: {len(ok)} ok, {len(rs) - len(ok)} failed, seeds {sorted({r['seed'] for r in ok})}")
        print(f"  hyperparameters: {rs[0]['hyperparameters']}")
        names = sorted({k for r in ok for k, v in r["metrics"].items() if isinstance(v, (int, float))})
        for m in names:
            v = [r["metrics"][m] for r in ok if m in r["metrics"]]
            mean = sum(v) / len(v)
            sd = math.sqrt(sum((x - mean) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else float("nan")
            print(f"  {m}: {mean:.4f} \u00b1 {sd:.4f} (n={len(v)})")
        for r in rs:
            if r["status"] != "ok":
                print(f"  FAILED {r['id']}: {r['error']}")


# ----------------------------------------------------------------- scaffold
T = {
    "README.md": """# {name}

## What
TODO: one paragraph: what this project is.

## Why
TODO: why it exists / what decision or question it informs.

## How to run
```bash
pip install -r requirements.txt
python train.py --seed 42        # reproduces the main result
python -m repro check            # audit against the checklist
```
Written for someone with no prior context. If a step needs asking you, add it here.
""",
    "RESEARCH_QUESTION.md": """# Project: {name}

## Question
TODO: one specific, answerable question (not a topic).

## Success metric
TODO: metric + threshold, written BEFORE any training. Must match metrics.py.

## Scope
TODO: time period, geography and/or population.

## Null or negative result plan
TODO: where a null/negative result or failure will be written up and shared.
""",
    "DATA_CARD.md": """# Data card: {name}

## Source and collection
TODO: where the data came from and how it was collected.

## Date range and version
TODO: e.g. collected 2024-01..2024-06; dataset version v1 (matches config data.version).

## Inclusion and exclusion criteria
TODO

## Licence
TODO: licence / terms of use, and the date you confirmed them.

## Preprocessing
TODO: link to the script(s) that do it (recorded in code, not just described).

## Columns and labels
TODO: name | type | unit | meaning. Define every label.

## Bias checks
TODO: what you checked (subgroup performance, label balance, sampling bias) and what you found.

## Re-identification risk
TODO: per-field risk assessment.

## Sharing plan
TODO: consent/licence basis; what is redacted, aggregated or synthetic.

## Data pointer
TODO: URL, DOI, or access instructions.

## Evaluation data
TODO: where the evaluation/test data lives (path or link) and its licence.
""",
    "MODEL_CARD.md": """# Model card: {name}

## Model details
TODO: name, version, authors, date, licence.

## Architecture
TODO: layers, sizes, parameter count; link to the model definition code.

## Intended use
TODO: intended and out-of-scope uses.

## Evaluation results
TODO: results on the evaluation data (metric from metrics.py, mean and std over seeds).

## Model parameters pointer
TODO: path or URL (Hugging Face / Zenodo DOI) of the FINAL checkpoint, and its licence.

## Intermediate checkpoints pointer
TODO: optional; URL of step/epoch checkpoints (Class I).

## Research paper
TODO: arXiv / DOI link, if any.

## Limitations
TODO
""",
    "LIMITATIONS.md": """# Limitations

## Limitation statement
TODO: the honest limitation a reviewer should know about.
""",
    "REPORT.md": """# Report

## Tuning procedure
TODO: grid size, folds, search budget, wall-clock time.

## Evaluation protocol
TODO: exact protocol; the reported metric must match it.

## Negative results and failure cases
TODO
""",
    "metrics.py": '''"""The metric definition lives here, in code."""


def accuracy(y_true, y_pred):
    """Fraction of exact matches."""
    return sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
''',
    "requirements.txt": "# pin your dependencies here (pip freeze > requirements.txt)\n",
    "train.py": '''"""Toy entry point. Replace the body with your real training code."""
import argparse, random
from repro import Run
from metrics import accuracy

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int)
args = ap.parse_args()

with Run("config.json", seed=args.seed) as run:
    hp = run.cfg["hyperparameters"]
    y = [random.randint(0, 1) for _ in range(200)]
    pred = [v if random.random() < 0.8 else 1 - v for v in y]  # fake model, ~80% right
    run.log_metric("val_accuracy", accuracy(y, pred))
    run.log_note(f"toy run with hyperparameters {hp}")
    # run.evaluate_test(lambda: accuracy(y_test, model.predict(X_test)))  # ONCE, at the very end
''',
}


def cmd_init(a):
    root = Path(a.name)
    root.mkdir(exist_ok=True)
    for fn, body in T.items():
        if not (root / fn).exists():
            (root / fn).write_text(body.format(name=a.name) if "{name}" in body else body)
    cfg = {"project": a.name, "seed": 42, "metric": "metrics.accuracy",
           "hyperparameters": {"lr": 0.1, "epochs": 5},
           "data": {"path": "data", "version": "v0"}, "entry_point": "train.py"}
    if not (root / "config.json").exists():
        (root / "config.json").write_text(json.dumps(cfg, indent=2))
    (root / "data").mkdir(exist_ok=True)
    (root / "data" / ".gitkeep").touch()
    (root / ".gitignore").write_text("__pycache__/\n.repro/\n")
    me = Path(__file__).resolve()
    if me != (root / "repro.py").resolve():
        (root / "repro.py").write_text(me.read_text())  # vendor the toolkit: no install needed
    print(f"created {root}/ . Next: cd {root}; git init; python train.py; python repro.py check")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="repro", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("init"); s.add_argument("name"); s.set_defaults(f=cmd_init)
    s = sub.add_parser("check"); s.add_argument("--core", action="store_true", help="shorter 'focus on results' subset"); s.set_defaults(f=cmd_check)
    s = sub.add_parser("mof", help="classify the project as MOF Class III / II / I and list what is missing")
    s.add_argument("--target", choices=["III", "II", "I"], default="I", help="exit non-zero if this class is not reached (default I)")
    s.set_defaults(f=cmd_mof)
    s = sub.add_parser("tick"); s.add_argument("ids", nargs="+"); s.set_defaults(f=cmd_tick)
    s = sub.add_parser("untick"); s.add_argument("ids", nargs="+"); s.set_defaults(f=lambda a: cmd_tick(a, False))
    s = sub.add_parser("decide"); s.add_argument("text"); s.set_defaults(f=cmd_decide)
    s = sub.add_parser("report"); s.set_defaults(f=cmd_report)
    a = ap.parse_args(argv)
    sys.exit(a.f(a) or 0)


if __name__ == "__main__":
    main()

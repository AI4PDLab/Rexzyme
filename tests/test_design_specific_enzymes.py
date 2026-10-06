"""Tests for src/design_specific_enzymes (DPO proxy + filtering pipeline).

Layout of the checks:

* structure   - files compile, scripts parse, paths/requirements are consistent
* wiring      - outputs of one script are the default inputs of the next
* behaviour   - the individual fixes (CLEAN parsing, CLEAN wrapper, empty metrics)
* integration - DPO script output -> filtering pipeline, on a tiny random model

External tools (CLEAN, AlphaFold, Rosetta, ...) are not run. CLEAN is replaced by a
fake package that mimics only its calling convention (paths relative to CLEAN/app).
"""

from __future__ import annotations

import ast
import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / "src" / "design_specific_enzymes"
SCRIPTS = DESIGN / "filtering_pipeline" / "scripts"
DPO = DESIGN / "dpo_proxy"
REL_SCRIPTS = "src/design_specific_enzymes/filtering_pipeline/scripts"

TARGET_EC = "1.17.1.9"
HUB_REPO = "AI4PD/REXzyme"


# --------------------------------------------------------------------------- helpers


def run(args, cwd, env=None):
    """Run a command with the current interpreter first on PATH and no GPU."""
    full_env = os.environ.copy()
    full_env["PATH"] = f"{Path(sys.executable).parent}{os.pathsep}{full_env['PATH']}"
    full_env["CUDA_VISIBLE_DEVICES"] = ""
    full_env.update(env or {})
    return subprocess.run(
        [str(a) for a in args], cwd=cwd, env=full_env, capture_output=True, text=True
    )


def md5(seq: str) -> str:
    return hashlib.md5(seq.encode("utf-8")).hexdigest()


def make_sequences(n: int) -> list[str]:
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    return [(alphabet[i % 20 :] + alphabet[: i % 20]) * 2 + alphabet[i % 7] * (i + 1) for i in range(n)]


def write_fasta(path: Path, records: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f">{h}\n{s}\n" for h, s in records), encoding="utf-8")


def dpo_style_records(n: int, ppls: list[float]) -> list[tuple[str, str]]:
    """FASTA records with the header format written by dpo_optimize_and_generate.py."""
    return [(f"{i}_ppl_{ppl:.4f}", seq) for i, (seq, ppl) in enumerate(zip(make_sequences(n), ppls))]


@pytest.fixture
def work(tmp_path: Path) -> Path:
    """Scratch dir shaped like the repo root: scripts use paths relative to it."""
    (tmp_path / "src").symlink_to(ROOT / "src")
    (tmp_path / "eval").symlink_to(ROOT / "eval")
    return tmp_path


@pytest.fixture(scope="session")
def hub_tokenizers() -> Path:
    """Real Rexzyme tokenizers (small); skip the tests that need them when offline."""
    try:
        from huggingface_hub import snapshot_download

        path = snapshot_download(HUB_REPO, allow_patterns=["tokenizer_aa/*", "tokenizer_smiles/*"])
    except Exception as exc:  # network / hub problems are not code failures
        pytest.skip(f"Hugging Face Hub not reachable: {exc}")
    return Path(path)


# --------------------------------------------------------------------------- structure

PY_FILES = sorted(DESIGN.rglob("*.py"))
SH_FILES = sorted(DESIGN.rglob("*.sh"))


def test_expected_files_are_present() -> None:
    assert len(PY_FILES) == 11
    assert len(SH_FILES) == 21
    assert (ROOT / "eval" / "clean" / "CLEAN_infer_fasta.py").is_file()


@pytest.mark.parametrize("path", PY_FILES, ids=lambda p: p.name)
def test_python_files_compile(path: Path) -> None:
    compile(path.read_text(encoding="utf-8"), str(path), "exec")


@pytest.mark.parametrize("path", SH_FILES, ids=lambda p: p.name)
def test_shell_scripts_have_valid_syntax(path: Path) -> None:
    result = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("path", PY_FILES, ids=lambda p: p.name)
def test_python_clis_show_help(path: Path, work: Path) -> None:
    result = run([sys.executable, path, "--help"], cwd=work)
    assert result.returncode == 0, result.stderr
    assert "usage" in result.stdout.lower()


def test_no_stale_references_to_the_old_folder_location() -> None:
    stale = re.compile(r"(?<![\w/])design_specific_enzymes/|\}/design_specific_enzymes/")
    offenders = []
    for path in DESIGN.rglob("*"):
        if path.is_file() and path.suffix in {".py", ".sh", ".md", ".yaml"}:
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if stale.search(line):
                    offenders.append(f"{path.relative_to(ROOT)}:{number}")
    assert not offenders, offenders


def test_every_referenced_script_exists() -> None:
    pattern = re.compile(r"src/design_specific_enzymes/[\w./-]+\.(?:py|sh)")
    missing = set()
    for path in DESIGN.rglob("*"):
        if path.is_file() and path.suffix in {".py", ".sh", ".md", ".yaml"}:
            for match in pattern.finditer(path.read_text(encoding="utf-8")):
                if not (ROOT / match.group(0)).is_file():
                    missing.add((path.name, match.group(0)))
    assert not missing, missing


def _requirement_names(*files: str) -> set[str]:
    names = set()
    for name in files:
        for line in (ROOT / name).read_text(encoding="utf-8").splitlines():
            line = line.split("#")[0].strip()
            if line:
                names.add(re.split(r"[<>=!~;\[ ]", line)[0].lower().replace("-", "_"))
    return names


def _third_party_imports(paths, local_dirs) -> set[str]:
    local = {p.stem for d in local_dirs for p in Path(d).glob("*.py")}
    modules = set()
    for path in paths:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                modules |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules.add(node.module.split(".")[0])
    return {m for m in modules if m not in sys.stdlib_module_names and m not in local}


IMPORT_TO_PACKAGE = {"Bio": "biopython"}


def test_design_specific_enzymes_imports_are_in_requirements() -> None:
    declared = _requirement_names("requirements.txt")
    needed = _third_party_imports([*PY_FILES, ROOT / "src" / "hf_assets.py"], [ROOT / "src"])
    needed = {IMPORT_TO_PACKAGE.get(m, m).lower() for m in needed}
    assert needed <= declared, f"not in requirements.txt: {sorted(needed - declared)}"


def test_xlsx_input_dependency_is_declared() -> None:
    # dpo_optimize_and_generate.read_table() accepts .xlsx through pandas.read_excel
    assert "openpyxl" in _requirement_names("requirements.txt")


def test_eval_imports_are_in_requirements() -> None:
    declared = _requirement_names("requirements.txt", "requirements-eval.txt")
    eval_files = [p for p in (ROOT / "eval").rglob("*.py")]
    local_dirs = [ROOT / "src", *{p.parent for p in eval_files}]
    needed = _third_party_imports(eval_files, local_dirs) - {"CLEAN"}  # CLEAN is an external tool
    needed = {IMPORT_TO_PACKAGE.get(m, m).lower() for m in needed}
    assert needed <= declared, f"not in requirements*.txt: {sorted(needed - declared)}"


# --------------------------------------------------------------------------- wiring


def _default(script: Path, variable: str) -> str:
    match = re.search(rf'{variable}="\$\{{{variable}:-([^}}]+)\}}"', script.read_text(encoding="utf-8"))
    assert match, f"{variable} default not found in {script.name}"
    return match.group(1)


def test_dpo_launcher_output_is_the_default_pipeline_input() -> None:
    produced = _default(DPO / "run_dpo_and_generate.sh", "FASTA_OUTPUT")
    consumers = ["run_deduplicate.sh", "run_metric_perplexity.sh", "run_metric_clean.sh"]
    for name in consumers:
        assert _default(SCRIPTS / name, "INPUT_FASTA") == produced, name
    for name in ("run_metric_alphafold.sh", "run_metric_esm1v.sh"):
        assert _default(SCRIPTS / name, "INPUT_FASTA") == produced, name


def test_stage_wrappers_chain_on_their_default_filenames() -> None:
    dedup_csv = _default(SCRIPTS / "run_deduplicate.sh", "OUTPUT_CSV")
    assert _default(SCRIPTS / "run_perplexity_filter.sh", "INPUT_CSV") == dedup_csv
    assert _default(SCRIPTS / "run_clean_filter.sh", "INPUT_CSV") == _default(
        SCRIPTS / "run_perplexity_filter.sh", "OUTPUT_CSV"
    )
    assert _default(SCRIPTS / "run_alphafold_filter.sh", "INPUT_CSV") == _default(
        SCRIPTS / "run_clean_filter.sh", "OUTPUT_CSV"
    )


def test_run_train_wrapper_uses_module_entrypoints() -> None:
    # `python src/tokenize.py` shadows the stdlib `tokenize` module and crashes on import.
    script = (ROOT / "scripts" / "run_train.sh").read_text(encoding="utf-8")
    assert "python -m src.tokenize" in script
    assert "python -m src.train" in script
    assert 'TOKENIZED_DIR="${TOKENIZED_DIR:-data/tokenized}"' in script


def test_run_train_wrapper_reports_missing_variables_cleanly(work: Path) -> None:
    result = run(["bash", ROOT / "scripts" / "run_train.sh"], cwd=work, env={"TOKENIZER_AA": ""})
    assert result.returncode != 0
    assert "unbound variable" not in result.stderr
    assert "TOKENIZER_AA" in result.stderr


def test_readme_does_not_pass_hub_subfolder_ids_to_local_only_entrypoints() -> None:
    readme = (DESIGN / "README.md").read_text(encoding="utf-8")
    assert not re.search(r"--tokenizer_(aa|mol)\s+AI4PD/", readme)
    assert "--model_name_or_path AI4PD" not in readme
    assert "snapshot_download" in readme and "hf_rexzyme/tokenizer_aa" in readme


def test_readme_tokenize_command_works_with_downloaded_tokenizers(
    work: Path, hub_tokenizers: Path
) -> None:
    data = work / "splits"
    data.mkdir()
    line = '{"translation": {"Reaction": "CC(=O)O>>CCO", "Sequence": "MKKLLAAGG"}}\n'
    (data / "train.jsonl").write_text(line * 4)
    (data / "val.jsonl").write_text(line * 2)
    result = run(
        [
            sys.executable, "-m", "src.tokenize",
            "--dataset", data,
            "--tokenizer_aa", hub_tokenizers / "tokenizer_aa",
            "--tokenizer_mol", hub_tokenizers / "tokenizer_smiles",
            "--output_dir", work / "tokenized",
        ],
        cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert (work / "tokenized" / "dataset_dict.json").is_file()


# --------------------------------------------------------------------------- behaviour: CLEAN parsing


def _clean_parse(work: Path, fasta_records, clean_text: str, target_ec: str = TARGET_EC):
    fasta = work / "in.fasta"
    write_fasta(fasta, fasta_records)
    clean = work / "clean.txt"
    clean.write_text(clean_text)
    out = work / "parsed.csv"
    result = run(
        [
            sys.executable, SCRIPTS / "parse_clean_predictions.py",
            "--clean_output", clean, "--input_fasta", fasta,
            "--output_csv", out, "--target_ec", target_ec,
        ],
        cwd=work,
    )
    return result, out


@pytest.mark.parametrize("separator", [" ", ","], ids=["whitespace", "comma"])
def test_clean_parser_maps_dpo_style_headers_to_sequences(work: Path, separator: str) -> None:
    records = dpo_style_records(3, [5.1, 3.2, 9.9])
    text = "".join(
        f"{header}{separator}EC:{ec}/0.1,EC:1.1.1.1/0.9\n"
        for (header, _), ec in zip(records, [TARGET_EC, "3.5.4.9", "2.7.7.7"])
    )
    result, out = _clean_parse(work, records, text)
    assert result.returncode == 0, result.stderr
    df = pd.read_csv(out, dtype=str)
    assert list(df["sequence_md5"]) == [md5(seq) for _, seq in records]
    assert list(df["predicted_ec"]) == [TARGET_EC, "3.5.4.9", "2.7.7.7"]
    assert set(df["target_ec"]) == {TARGET_EC}


def test_clean_parser_header_match_beats_trailing_number_index(work: Path) -> None:
    # Header "0_ppl_2" ends in "2", which would be misread as record index 2.
    records = [("0_ppl_2", "AAAA"), ("1_ppl_0", "CCCC"), ("2_ppl_1", "DDDD")]
    result, out = _clean_parse(work, records, "0_ppl_2 EC:1.1.1.1/0.1\n")
    assert result.returncode == 0, result.stderr
    assert list(pd.read_csv(out)["sequence_md5"]) == [md5("AAAA")]


@pytest.mark.parametrize("sample_ids", [["0", "1"], ["file_0", "file_1"]], ids=["plain-int", "suffix-int"])
def test_clean_parser_still_supports_index_style_ids(work: Path, sample_ids) -> None:
    records = [("seqA", "AAAA"), ("seqB", "CCCC")]
    text = "".join(f"{sid} EC:1.1.1.{i}/0.1\n" for i, sid in enumerate(sample_ids, 1))
    result, out = _clean_parse(work, records, text)
    assert result.returncode == 0, result.stderr
    df = pd.read_csv(out, dtype=str)
    assert list(df["sequence_md5"]) == [md5("AAAA"), md5("CCCC")]
    assert list(df["predicted_ec"]) == ["1.1.1.1", "1.1.1.2"]


def test_clean_parser_fails_loudly_when_nothing_matches(work: Path) -> None:
    records = dpo_style_records(2, [1.0, 2.0])
    result, out = _clean_parse(work, records, "unrelated_name EC:1.1.1.1/0.1\nother EC:1.1.1.2/0.1\n")
    assert result.returncode != 0
    assert "no CLEAN predictions" in result.stderr
    assert not out.exists()


def test_clean_parser_warns_on_partial_match(work: Path) -> None:
    records = dpo_style_records(2, [1.0, 2.0])
    text = f"{records[0][0]} EC:1.1.1.1/0.1\nunknown EC:1.1.1.2/0.1\n"
    result, out = _clean_parse(work, records, text)
    assert result.returncode == 0
    assert "1 CLEAN line(s)" in result.stderr
    assert len(pd.read_csv(out)) == 1


def test_clean_parser_ignores_ambiguous_headers(work: Path) -> None:
    # two records share a first token: must not silently pick one of them
    records = [("dup x", "AAAA"), ("dup y", "CCCC")]
    result, _ = _clean_parse(work, records, "dup EC:1.1.1.1/0.1\n")
    assert result.returncode != 0


# --------------------------------------------------------------------------- behaviour: CLEAN wrapper


@pytest.fixture
def fake_clean(tmp_path: Path) -> Path:
    """A stand-in for CLEAN that only enforces its calling convention.

    Like the real tool it reads/writes paths relative to the current directory, so the
    wrapper must run it from CLEAN/app. Output spelling is chosen via FAKE_CLEAN_OUT.
    """
    pkg = tmp_path / "fakeclean" / "CLEAN"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "utils.py").write_text(
        "import os\n"
        "def prepare_infer_fasta(test_data):\n"
        "    assert os.path.exists(f'data/{test_data}.fasta'), 'not run from CLEAN/app: ' + os.getcwd()\n"
        "    open(f'data/{test_data}.csv', 'w').write('tmp')\n"
    )
    (pkg / "infer.py").write_text(
        "import os\n"
        "def infer_maxsep(train_data, test_data, report_metrics, pretrained, gmm):\n"
        "    assert train_data == 'split100'\n"
        "    stem = test_data.split('/', 1)[1]\n"
        "    ids = [l[1:].strip() for l in open(f'data/{test_data}.fasta') if l.startswith('>')]\n"
        "    os.makedirs('results/inputs', exist_ok=True)\n"
        "    name = os.environ['FAKE_CLEAN_OUT'].format(stem=stem)\n"
        "    with open(f'results/inputs/{name}', 'w') as f:\n"
        "        for h in ids:\n"
        "            f.write(f'{h},EC:1.17.1.9/0.1,EC:1.1.1.1/0.9\\n')\n"
    )
    app = tmp_path / "CLEAN" / "app"
    app.mkdir(parents=True)
    return tmp_path


@pytest.mark.parametrize("out_name", ["{stem}.csv", "{stem}_maxsep.csv"])
def test_clean_wrapper_runs_clean_from_its_app_dir_and_parses_output(
    work: Path, fake_clean: Path, out_name: str
) -> None:
    records = dpo_style_records(3, [4.0, 5.0, 6.0])
    write_fasta(work / "results" / "dpo_generated.fasta", records)
    result = run(
        ["bash", SCRIPTS / "run_metric_clean.sh"],
        cwd=work,
        env={
            "CLEAN_APP_DIR": str(fake_clean / "CLEAN" / "app"),
            "PYTHONPATH": str(fake_clean / "fakeclean"),
            "FAKE_CLEAN_OUT": out_name,
            "TARGET_EC": TARGET_EC,
        },
    )
    assert result.returncode == 0, result.stderr
    df = pd.read_csv(work / "results" / "filtering" / "clean_scores.csv", dtype=str)
    assert list(df["sequence_md5"]) == [md5(seq) for _, seq in records]
    assert set(df["predicted_ec"]) == {TARGET_EC}


def test_clean_wrapper_rejects_placeholder_app_dir(work: Path) -> None:
    write_fasta(work / "results" / "dpo_generated.fasta", dpo_style_records(2, [1.0, 2.0]))
    result = run(["bash", SCRIPTS / "run_metric_clean.sh"], cwd=work)
    assert result.returncode == 1
    assert "CLEAN_APP_DIR" in result.stderr


def test_clean_wrapper_fails_when_clean_wrote_no_output(work: Path, fake_clean: Path) -> None:
    write_fasta(work / "results" / "dpo_generated.fasta", dpo_style_records(2, [1.0, 2.0]))
    result = run(
        ["bash", SCRIPTS / "run_metric_clean.sh"],
        cwd=work,
        env={
            "CLEAN_APP_DIR": str(fake_clean / "CLEAN" / "app"),
            "PYTHONPATH": str(fake_clean / "fakeclean"),
            "FAKE_CLEAN_OUT": "somewhere_else.csv",
        },
    )
    assert result.returncode == 1
    assert "no CLEAN output found" in result.stderr


# --------------------------------------------------------------------------- behaviour: empty metrics


def test_candidate_builder_warns_when_fasta_has_no_perplexity(work: Path) -> None:
    fasta = work / "plain.fasta"
    write_fasta(fasta, [("0", "AAAA"), ("1", "CCCC")])  # headers as written by src/inference.py
    out = work / "cand.csv"
    result = run(
        [sys.executable, SCRIPTS / "build_candidates_from_fasta.py",
         "--input_fasta", fasta, "--output_csv", out],
        cwd=work,
    )
    assert result.returncode == 0
    assert "perplexity" in result.stderr and "WARNING" in result.stderr


def test_candidate_builder_is_quiet_when_perplexity_is_present(work: Path) -> None:
    fasta = work / "dpo.fasta"
    write_fasta(fasta, dpo_style_records(2, [1.5, 2.5]))
    out = work / "cand.csv"
    result = run(
        [sys.executable, SCRIPTS / "build_candidates_from_fasta.py",
         "--input_fasta", fasta, "--output_csv", out],
        cwd=work,
    )
    assert result.returncode == 0
    assert "WARNING" not in result.stderr
    assert list(pd.read_csv(out)["perplexity"]) == [1.5, 2.5]


@pytest.mark.parametrize(
    "script,extra",
    [("filter_top_fraction.py", ["--fraction", "0.5"]), ("filter_top_n.py", ["--top_n", "2"])],
)
def test_top_filters_fail_when_metric_column_is_empty(work: Path, script: str, extra) -> None:
    table = work / "t.csv"
    pd.DataFrame({"sequence_md5": ["a", "b"], "perplexity": [None, None]}).to_csv(table, index=False)
    result = run(
        [sys.executable, SCRIPTS / script, "--input_csv", table, "--output_csv", work / "o.csv",
         "--column", "perplexity", "--direction", "lower_is_better", *extra],
        cwd=work,
    )
    assert result.returncode != 0
    assert "no numeric values" in result.stderr


@pytest.mark.parametrize(
    "script,extra",
    [("filter_top_fraction.py", ["--fraction", "0.5"]), ("filter_top_n.py", ["--top_n", "2"])],
)
def test_top_filters_accept_an_already_empty_table(work: Path, script: str, extra) -> None:
    table = work / "t.csv"
    pd.DataFrame({"sequence_md5": [], "perplexity": []}).to_csv(table, index=False)
    result = run(
        [sys.executable, SCRIPTS / script, "--input_csv", table, "--output_csv", work / "o.csv",
         "--column", "perplexity", "--direction", "lower_is_better", *extra],
        cwd=work,
    )
    assert result.returncode == 0, result.stderr
    assert len(pd.read_csv(work / "o.csv")) == 0


# --------------------------------------------------------------------------- integration: filtering chain


def test_steps_1_to_3_chain_on_default_filenames(work: Path) -> None:
    """dpo-style FASTA -> dedupe -> perplexity top 25% -> CLEAN parse -> merge -> EC filter."""
    ppls = [8.0, 3.0, 9.0, 1.0, 7.0, 6.0, 5.0, 4.0]
    records = dpo_style_records(8, ppls)
    records.append(("8_ppl_0.5000", records[0][1]))  # exact duplicate of record 0
    write_fasta(work / "results" / "dpo_generated.fasta", records)
    env = {"TARGET_EC": TARGET_EC}

    step1 = run(["bash", SCRIPTS / "run_deduplicate.sh"], cwd=work, env=env)
    assert step1.returncode == 0, step1.stderr
    base = pd.read_csv(work / "results" / "step01_deduplicated.csv")
    assert len(base) == 8 and base["perplexity"].notna().all()

    step2 = run(["bash", SCRIPTS / "run_perplexity_filter.sh"], cwd=work)
    assert step2.returncode == 0, step2.stderr
    top = pd.read_csv(work / "results" / "step02_perplexity_top25.csv")
    assert sorted(top["perplexity"]) == [1.0, 3.0]  # 25% of 8, lowest perplexity

    # CLEAN says the lowest-perplexity sequence is the target EC and the other is not.
    best, other = top.sort_values("perplexity")["header"]
    clean_out = work / "clean_raw.txt"
    clean_out.write_text(f"{best},EC:{TARGET_EC}/0.1\n{other},EC:3.5.4.9/0.1\n")
    parsed = run(
        [sys.executable, SCRIPTS / "parse_clean_predictions.py",
         "--clean_output", clean_out, "--input_fasta", work / "results" / "step01_deduplicated.fasta",
         "--output_csv", work / "clean_scores.csv"],
        cwd=work,
    )
    assert parsed.returncode == 0, parsed.stderr
    merged = run(
        [sys.executable, SCRIPTS / "merge_metric_table.py",
         "--base_csv", work / "results" / "step02_perplexity_top25.csv",
         "--metric_csv", work / "clean_scores.csv",
         "--output_csv", work / "results" / "step02_with_clean.csv",
         "--columns", "predicted_ec"],
        cwd=work,
    )
    assert merged.returncode == 0, merged.stderr

    step3 = run(
        ["bash", SCRIPTS / "run_clean_filter.sh"],
        cwd=work,
        env={"INPUT_CSV": "results/step02_with_clean.csv"},
    )
    assert step3.returncode == 0, step3.stderr
    final = pd.read_csv(work / "results" / "step03_clean_ec_match.csv")
    assert list(final["header"]) == [best]


def test_perplexity_filter_stops_for_fasta_without_perplexity(work: Path) -> None:
    write_fasta(work / "results" / "dpo_generated.fasta", [(str(i), s) for i, s in enumerate(make_sequences(4))])
    assert run(["bash", SCRIPTS / "run_deduplicate.sh"], cwd=work).returncode == 0
    result = run(["bash", SCRIPTS / "run_perplexity_filter.sh"], cwd=work)
    assert result.returncode != 0
    assert "no numeric values" in result.stderr


def test_full_chain_script_on_a_merged_table(work: Path) -> None:
    import random

    rng = random.Random(0)
    n = 2000
    table = pd.DataFrame(
        {
            "sequence_md5": [f"h{i}" for i in range(n)],
            "perplexity": [rng.uniform(1, 20) for _ in range(n)],
            "predicted_ec": [rng.choice([TARGET_EC, TARGET_EC, "1.1.1.1"]) for _ in range(n)],
            "target_ec": [TARGET_EC] * n,
            "plddt": [rng.uniform(50, 95) for _ in range(n)],
            "identity_to_natural": [rng.uniform(0.3, 1.0) for _ in range(n)],
            "iptm": [rng.random() for _ in range(n)],
            "tmscore": [rng.random() for _ in range(n)],
            "esm1v_score": [-rng.uniform(1, 10) for _ in range(n)],
            "proteinmpnn_score": [-rng.uniform(0, 3) for _ in range(n)],
            "rosetta_score": [-rng.uniform(100, 900) for _ in range(n)],
            "relative_hydrophobic_sasa": [rng.uniform(0, 4) for _ in range(n)],
            "net_charge": [rng.choice([-3, -1, 0, 2, 5]) for _ in range(n)],
            "average_sap": [rng.uniform(0, 0.8) for _ in range(n)],
            "hydrophobic_sasa_to_ideal_ratio": [rng.uniform(0.5, 2.2) for _ in range(n)],
        }
    )
    table.to_csv(work / "merged.csv", index=False)
    result = run(
        ["bash", SCRIPTS / "run_filtering_pipeline.sh"],
        cwd=work,
        env={"IN": "merged.csv", "OUT_DIR": "out"},
    )
    assert result.returncode == 0, result.stderr

    out = work / "out"
    step02 = pd.read_csv(out / "step02_perplexity_top25.csv")
    assert len(step02) == n // 4
    assert step02["perplexity"].max() <= table["perplexity"].sort_values().iloc[n // 4 - 1] + 1e-9

    step03 = pd.read_csv(out / "step03_clean_ec_match.csv")
    assert (step03["predicted_ec"] == TARGET_EC).all()

    final = pd.read_csv(out / "step11_rosetta_final.csv")
    assert 0 < len(final) < len(step03)
    assert (final["relative_hydrophobic_sasa"] <= 3.0).all()
    assert (final["average_sap"] <= 0.5).all()
    assert (final["hydrophobic_sasa_to_ideal_ratio"] <= 1.7).all()
    assert (final["net_charge"] != 0).all()
    assert final["rosetta_score"].is_monotonic_increasing
    assert (final["plddt"] > 75).all()


# --------------------------------------------------------------------------- integration: DPO -> pipeline


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory) -> Path:
    """Randomly initialised T5 sized for the real Rexzyme tokenizers (not a trained model)."""
    transformers = pytest.importorskip("transformers")
    from huggingface_hub import snapshot_download

    try:
        tok_dir = Path(
            snapshot_download(HUB_REPO, allow_patterns=["tokenizer_aa/*", "tokenizer_smiles/*"])
        )
    except Exception as exc:
        pytest.skip(f"Hugging Face Hub not reachable: {exc}")
    aa = transformers.AutoTokenizer.from_pretrained(tok_dir / "tokenizer_aa")
    mol = transformers.AutoTokenizer.from_pretrained(tok_dir / "tokenizer_smiles")
    config = transformers.T5Config(
        vocab_size=max(len(aa), len(mol)),
        d_model=32, d_ff=64, d_kv=8, num_heads=2, num_layers=1, num_decoder_layers=1,
        decoder_start_token_id=aa.pad_token_id, pad_token_id=aa.pad_token_id,
        eos_token_id=aa.eos_token_id,
    )
    path = tmp_path_factory.mktemp("tiny_model")
    transformers.T5ForConditionalGeneration(config).save_pretrained(path)
    return path


def _dpo_args(model: Path, out_dir: Path, mode: str, fasta: Path, extra=()):
    return [
        sys.executable, DPO / "dpo_optimize_and_generate.py",
        "--model_path", model, "--output_model_dir", out_dir,
        "--tokenizer_aa", f"{HUB_REPO}/tokenizer_aa",  # 3-part Hub id, as in the launcher
        "--tokenizer_mol", f"{HUB_REPO}/tokenizer_smiles",
        "--activity_file", DPO / "example_proxy_input.csv",
        "--proxy_column", "Kcat/Km", "--reaction", "CC(=O)O>>CCO", "--mode", mode,
        "--num_epochs", "1", "--batch_size", "2", "--max_target_length", "64",
        "--generate_after_dpo", "--generation_rounds", "1", "--num_return_sequences", "4",
        "--generation_max_length", "40", "--fasta_output", fasta, *extra,
    ]


@pytest.mark.parametrize("mode", ["ranked", "weighted"])
def test_dpo_output_feeds_the_filtering_pipeline(work: Path, tiny_model: Path, mode: str) -> None:
    fasta = work / "results" / "dpo_generated.fasta"
    result = run(_dpo_args(tiny_model, work / "dpo_model", mode, fasta), cwd=work)
    assert result.returncode == 0, result.stderr[-3000:]

    assert (work / "dpo_model" / "config.json").is_file()
    assert (work / "dpo_model" / "tokenizer_aa").is_dir()
    assert (work / "dpo_model" / "tokenizer_smiles").is_dir()
    headers = [l[1:] for l in fasta.read_text().splitlines() if l.startswith(">")]
    assert len(headers) == 4
    assert all(re.fullmatch(r"\d+_ppl_\d+\.\d+", h) for h in headers), headers

    # producer -> consumers: perplexity must survive into the candidate table
    step1 = run(["bash", SCRIPTS / "run_deduplicate.sh"], cwd=work)
    assert step1.returncode == 0, step1.stderr
    table = pd.read_csv(work / "results" / "step01_deduplicated.csv")
    assert len(table) >= 1 and table["perplexity"].notna().all()
    assert "WARNING" not in step1.stderr
    step2 = run(["bash", SCRIPTS / "run_perplexity_filter.sh"], cwd=work)
    assert step2.returncode == 0, step2.stderr

    # and CLEAN output keyed by those headers is understood
    clean = work / "clean_raw.txt"
    clean.write_text("".join(f"{h} EC:{TARGET_EC}/0.1\n" for h in headers))
    parsed = run(
        [sys.executable, SCRIPTS / "parse_clean_predictions.py", "--clean_output", clean,
         "--input_fasta", fasta, "--output_csv", work / "clean_scores.csv"],
        cwd=work,
    )
    assert parsed.returncode == 0, parsed.stderr
    assert len(pd.read_csv(work / "clean_scores.csv")) == len(headers)


def test_dpo_launcher_writes_where_the_pipeline_reads(work: Path, tiny_model: Path) -> None:
    result = run(
        ["bash", DPO / "run_dpo_and_generate.sh"],
        cwd=work,
        env={
            "MODEL_PATH": str(tiny_model),
            "OUTPUT_MODEL_DIR": "results/dpo_model",
            "ACTIVITY_FILE": str(DPO / "example_proxy_input.csv"),
            "REACTION": "CC(=O)O>>CCO",
            "BATCH_SIZE": "2",
            "NUM_EPOCHS": "1",
            "GENERATION_ROUNDS": "1",
            "NUM_RETURN_SEQUENCES": "3",
        },
    )
    assert result.returncode == 0, result.stderr[-3000:]
    fasta = work / "results" / "dpo_generated.fasta"
    assert fasta.is_file()
    assert run(["bash", SCRIPTS / "run_deduplicate.sh"], cwd=work).returncode == 0

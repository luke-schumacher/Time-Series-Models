"""Tests for the stale-/tmp-copy guard used by steps 05 and 06.

Regression: on 2026-07-27 step 06 died with a bare
`ModuleNotFoundError: No module named 'AlternatingPipeline.models.checkpoint_compat'`
because TMP_ROOT had been copied by step 04 during the 07-24 training run,
before that module existed, on a cluster that had been up ever since. The error
named the module, not the stale copy, which is the wrong thing to look at.
"""
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "DatabricksPipeline", "csv_pipeline_seqparams", "config.py",
)


@pytest.fixture(scope="module")
def seqparams_config():
    """Load the notebook-style config as a module (pure constants + funcs)."""
    module = types.ModuleType("seqparams_config_under_test")
    with open(CONFIG_PATH) as handle:
        exec(compile(handle.read(), CONFIG_PATH, "exec"), module.__dict__)
    return module


def _make_source_tree(root, dotted_modules):
    for dotted in dotted_modules:
        path = os.path.join(root, *dotted.split(".")) + ".py"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as handle:
            handle.write("# stub\n")


def test_passes_when_every_required_module_is_present(seqparams_config, tmp_path):
    required = [
        "AlternatingPipeline.config",
        "AlternatingPipeline.models.checkpoint_compat",
    ]
    _make_source_tree(str(tmp_path), required)

    assert seqparams_config.assert_pipeline_source_fresh(
        str(tmp_path), required_modules=required, purge=False
    ) is True


def test_raises_naming_the_missing_module_and_the_remedy(seqparams_config, tmp_path):
    _make_source_tree(str(tmp_path), ["AlternatingPipeline.config"])

    with pytest.raises(RuntimeError) as excinfo:
        seqparams_config.assert_pipeline_source_fresh(
            str(tmp_path),
            required_modules=[
                "AlternatingPipeline.config",
                "AlternatingPipeline.models.checkpoint_compat",
            ],
            purge=False,
        )

    message = str(excinfo.value)
    assert "AlternatingPipeline.models.checkpoint_compat" in message
    assert "AlternatingPipeline.config" not in message.split("missing:")[1].split("\n")[0]
    # Must point at the stale copy and the fix, not just the symptom.
    assert str(tmp_path) in message
    assert "04_train_models.py" in message
    assert "do NOT run the training cell" in message


def test_absent_directory_is_not_diagnosed_as_a_stale_copy(seqparams_config, tmp_path):
    """The 2026-08-21 regression, and why the two messages must differ.

    Steps 05 and 06 both died reporting EVERY required module missing, after a
    step-04 run that had itself succeeded. The guard said "copied before those
    modules existed", which sent the fix toward a git pull — but /tmp is
    per-machine, step 04 had run on a GPU cluster, and the analysis had not.
    Nothing had been copied there at all, and no pull could have helped.

    A STALE copy is missing SOME modules. An ABSENT one is missing all of them
    because it was never made here. Those need different remedies, so they need
    different messages.
    """
    absent = str(tmp_path / "never_copied")

    with pytest.raises(RuntimeError) as excinfo:
        seqparams_config.assert_pipeline_source_fresh(
            absent, required_modules=["AlternatingPipeline.config"], purge=False
        )

    message = str(excinfo.value)
    assert "Stale source copy" not in message
    assert "does not exist" in message
    assert "bootstrap_pipeline_source()" in message
    assert absent in message


def test_present_but_empty_directory_is_not_diagnosed_as_a_stale_copy(
    seqparams_config, tmp_path
):
    """A directory that exists but holds none of the modules is not stale either.

    Staleness is defined by what a copy PREDATES, so a copy missing everything
    predates nothing — it is an empty or foreign directory, and telling the
    reader to pull and re-copy would be guessing.
    """
    empty = tmp_path / "empty_but_present"
    empty.mkdir()

    with pytest.raises(RuntimeError) as excinfo:
        seqparams_config.assert_pipeline_source_fresh(
            str(empty),
            required_modules=[
                "AlternatingPipeline.config",
                "AlternatingPipeline.models.checkpoint_compat",
            ],
            purge=False,
        )

    message = str(excinfo.value)
    assert "Stale source copy" not in message
    assert "contains none of" in message
    assert "bootstrap_pipeline_source()" in message


def test_purge_evicts_stale_namespace_packages(seqparams_config, tmp_path):
    """Top-level namespace packages have __file__ is None; purge by name."""
    required = ["AlternatingPipeline.config"]
    _make_source_tree(str(tmp_path), required)

    stale_pkg = types.ModuleType("AlternatingPipeline")
    stale_pkg.__file__ = None  # namespace package, as in Databricks
    sys.modules["AlternatingPipeline"] = stale_pkg
    sys.modules["AlternatingPipeline.models"] = types.ModuleType("AlternatingPipeline.models")
    sys.modules["csv_pipeline_seqparams.config"] = types.ModuleType("csv_pipeline_seqparams.config")
    sys.modules["unrelated_module"] = types.ModuleType("unrelated_module")

    try:
        seqparams_config.assert_pipeline_source_fresh(
            str(tmp_path), required_modules=required, purge=True
        )

        assert "AlternatingPipeline" not in sys.modules
        assert "AlternatingPipeline.models" not in sys.modules
        assert "csv_pipeline_seqparams.config" not in sys.modules
        assert "unrelated_module" in sys.modules, "purge must not touch other modules"
    finally:
        sys.modules.pop("unrelated_module", None)


def test_purge_evicts_legacy_top_level_modules_by_file_path(seqparams_config, tmp_path):
    """The 2026-07-27 regression: name-prefix matching alone is not enough.

    AlternatingPipeline/models/examination_model.py does
    `from models.sequence_generator import ...` — a legacy TOP-LEVEL name that
    no prefix in ("AlternatingPipeline", "csv_pipeline_seqparams") matches. It
    survived a re-run in a persistent kernel, so a fresh sha in the pre-flight
    sat alongside a stale model class, detectable only via the parameter count.
    """
    required = ["AlternatingPipeline.config"]
    _make_source_tree(str(tmp_path), required)

    legacy = types.ModuleType("models.sequence_generator")
    legacy.__file__ = os.path.join(str(tmp_path), "AlternatingPipeline", "models",
                                   "sequence_generator.py")
    sys.modules["models.sequence_generator"] = legacy
    sys.modules["config"] = types.ModuleType("config")
    sys.modules["config"].__file__ = os.path.join(str(tmp_path), "AlternatingPipeline", "config.py")

    elsewhere = types.ModuleType("elsewhere")
    elsewhere.__file__ = "/usr/lib/python3/elsewhere.py"
    sys.modules["elsewhere"] = elsewhere

    try:
        seqparams_config.assert_pipeline_source_fresh(
            str(tmp_path), required_modules=required, purge=True
        )

        assert "models.sequence_generator" not in sys.modules
        assert "config" not in sys.modules
        assert "elsewhere" in sys.modules, "purge must not evict modules outside tmp_root"
    finally:
        for name in ("models.sequence_generator", "config", "elsewhere"):
            sys.modules.pop(name, None)


def test_no_required_modules_is_a_purge_only_noop(seqparams_config, tmp_path):
    assert seqparams_config.assert_pipeline_source_fresh(str(tmp_path)) is True


# ---------------------------------------------------------------------------
# bootstrap_pipeline_source — the replacement for all of the above.
#
# The guard tests above are about diagnosing a bad /tmp copy. This section is
# about not needing one: a notebook that depends on another notebook's side
# effect in /tmp is not reproducible, and on 2026-08-21 that cost a finished
# GPU training run its entire downstream analysis.
# ---------------------------------------------------------------------------

def _fake_repo(root):
    """A directory that passes is_healthy_repo (config.py must be READABLE)."""
    pkg = os.path.join(root, "AlternatingPipeline")
    os.makedirs(pkg, exist_ok=True)
    with open(os.path.join(pkg, "config.py"), "w") as handle:
        handle.write("# stub\n")
    return root


def test_health_check_requires_readable_bytes_not_just_existence(
    seqparams_config, tmp_path
):
    """The Shared Workspace mount answers os.path.isfile and then raises
    OSError on read, so the check has to actually touch bytes."""
    assert seqparams_config.is_healthy_repo(str(tmp_path)) is False
    assert seqparams_config.is_healthy_repo(_fake_repo(str(tmp_path))) is True


def test_the_first_healthy_candidate_wins_and_nothing_is_cloned(
    seqparams_config, tmp_path
):
    good = _fake_repo(str(tmp_path / "good"))
    calls = []

    root = seqparams_config.bootstrap_pipeline_source(
        candidates=(str(tmp_path / "absent"), good),
        refresh=False, verbose=False, preimport=(),
        runner=lambda cmd: calls.append(cmd),
    )

    assert root == good
    assert calls == [], "a healthy candidate must not trigger a clone"


def test_no_healthy_candidate_clones_into_the_tmp_path(seqparams_config, tmp_path):
    clone_target = str(tmp_path / "tsm")

    def _runner(cmd):
        assert cmd[:2] == ["git", "clone"]
        _fake_repo(clone_target)

    root = seqparams_config.bootstrap_pipeline_source(
        candidates=(str(tmp_path / "absent"),),
        tmp_clone=clone_target, refresh=False, verbose=False, preimport=(),
        runner=_runner,
    )
    assert root == clone_target


def test_a_failed_refresh_warns_instead_of_stopping(seqparams_config, tmp_path):
    """An offline cluster must still be able to analyse a checkpoint with the
    source it already has. The commit it is running is printed either way."""
    clone = _fake_repo(str(tmp_path / "tsm"))
    os.makedirs(os.path.join(clone, ".git"), exist_ok=True)

    def _runner(cmd):
        raise OSError("no network")

    root = seqparams_config.bootstrap_pipeline_source(
        candidates=(clone,), tmp_clone=clone,
        refresh=True, verbose=False, preimport=(), runner=_runner,
    )
    assert root == clone


def test_the_broken_workspace_mount_is_scrubbed_from_sys_path(
    seqparams_config, tmp_path
):
    poisoned = "/Workspace/Shared/Patient Exchange and Examination/Time-Series-Models"
    sys.path.insert(0, poisoned)
    try:
        seqparams_config.bootstrap_pipeline_source(
            candidates=(_fake_repo(str(tmp_path)),),
            refresh=False, verbose=False, preimport=(), runner=lambda cmd: None,
        )
        assert poisoned not in sys.path
    finally:
        while poisoned in sys.path:
            sys.path.remove(poisoned)


def test_every_serving_notebook_uses_the_shared_bootstrap(seqparams_config):
    """Three notebooks used to answer "where does source come from?" three
    different ways, and the two that could not stand alone are the two that
    died. A private re-implementation is how the shared one stops matching.
    """
    pipeline_dir = os.path.dirname(CONFIG_PATH)
    for name in ("05_feature_analysis.py", "06_compare_models.py",
                 "07_generate_synthetic_data.py"):
        with open(os.path.join(pipeline_dir, name)) as handle:
            source = handle.read()
        # Comments are stripped: these files explain the old mechanism at
        # length, and matching prose would make the assertion meaningless.
        code = "\n".join(line for line in source.splitlines()
                         if not line.lstrip().startswith("#"))

        assert "bootstrap_pipeline_source()" in code, name
        assert "_REPO_CANDIDATES" not in code, f"{name} re-grew a private bootstrap"
        assert "_is_healthy_repo" not in code, f"{name} re-grew a private bootstrap"
        assert "/tmp/alternating_pipeline_src" not in code, name
        assert "/tmp/tsm" not in code, f"{name} hardcodes the clone path again"
        # The bootstrap's refresh WARNS rather than fails when git cannot reach
        # the network, so "it returned a root" does not mean "that root is
        # current". The post-condition is what makes an unrefreshed clone a
        # readable error instead of a ModuleNotFoundError deep in the run.
        assert "assert_pipeline_source_fresh(REPO_ROOT" in code, name

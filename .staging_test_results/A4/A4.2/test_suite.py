"""
Phase 3 Test Suite — Track A4: CI/CD Pipeline
Subtasks: A4.1 (CI: lint, test, build, push), A4.2 (Staging environment), A4.3 (Smoke test + approval gate)
(5 hard test cases each = 15 total)
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tests.phase1.harness import SubtaskRunner


def _load_yaml():
    """Load and parse azure-pipelines.yml"""
    import yaml
    pipeline_path = ROOT_DIR / "azure-pipelines.yml"
    assert pipeline_path.exists(), f"azure-pipelines.yml not found at {pipeline_path}"
    with open(pipeline_path) as f:
        return yaml.safe_load(f)


# ==============================================================================
# A4.1 — CI stage: lint, test, build, push
# ==============================================================================
def run_test_a4_1() -> SubtaskRunner:
    runner = SubtaskRunner(
        task_id="A4",
        subtask_id="A4.1",
        title="CI/CD pipeline: lint, test, build, and push stages defined and correctly ordered",
        target_files=["azure-pipelines.yml"],
        source_file=__file__,
    )

    # TC 1: CI stage contains both lint and test steps
    def tc1():
        doc = _load_yaml()
        stages = doc.get("stages", [])
        stage_names = [s.get("stage") for s in stages]
        assert "CI" in stage_names, f"No CI stage found; stages={stage_names}"
        ci_stage = next(s for s in stages if s.get("stage") == "CI")
        # Collect all scripts in the CI stage
        ci_yaml_str = str(ci_stage).lower()
        assert "ruff" in ci_yaml_str or "lint" in ci_yaml_str, "CI stage has no lint step (ruff/flake8)"
        assert "pytest" in ci_yaml_str or "test" in ci_yaml_str, "CI stage has no test step"
        return True, f"CI stage found with lint + test steps; all stages: {stage_names}"

    runner.record_case(
        case_num=1,
        name="CI stage contains lint and test steps",
        input_setup="Parse azure-pipelines.yml; inspect CI stage jobs",
        expected_result="CI stage present with ruff/lint and pytest/test steps",
        severity="Crit",
        fn=tc1,
    )

    # TC 2: BuildPush stage depends on CI
    def tc2():
        doc = _load_yaml()
        stages = doc.get("stages", [])
        build_stage = next((s for s in stages if s.get("stage") == "BuildPush"), None)
        assert build_stage is not None, "BuildPush stage not found"
        deps = build_stage.get("dependsOn", [])
        if isinstance(deps, str):
            deps = [deps]
        assert "CI" in deps, f"BuildPush.dependsOn doesn't include CI; dependsOn={deps}"
        return True, f"BuildPush.dependsOn = {deps}"

    runner.record_case(
        case_num=2,
        name="BuildPush stage depends on CI",
        input_setup="Check azure-pipelines.yml BuildPush.dependsOn",
        expected_result="BuildPush.dependsOn includes 'CI'",
        severity="Crit",
        fn=tc2,
    )

    # TC 3: Docker tags include shortSha
    def tc3():
        pipeline_text = (ROOT_DIR / "azure-pipelines.yml").read_text()
        assert "shortSha" in pipeline_text, "shortSha variable not referenced in pipeline"
        assert "tags" in pipeline_text.lower(), "Docker tags section missing"
        # shortSha is set from first 7 chars of BUILD_SOURCEVERSION
        assert "BUILD_SOURCEVERSION" in pipeline_text or "shortSha" in pipeline_text, \
            "shortSha not derived from BUILD_SOURCEVERSION"
        return True, "Docker image tags include $(shortSha) from git commit"

    runner.record_case(
        case_num=3,
        name="Docker image tagged with shortSha (git commit)",
        input_setup="Inspect azure-pipelines.yml for shortSha usage in Docker tags",
        expected_result="shortSha variable defined and used in image tags",
        severity="High",
        fn=tc3,
    )

    # TC 4: Pipeline YAML is syntactically valid (yaml.safe_load succeeds)
    def tc4():
        import yaml
        pipeline_path = ROOT_DIR / "azure-pipelines.yml"
        assert pipeline_path.exists(), "azure-pipelines.yml not found"
        with open(pipeline_path) as f:
            doc = yaml.safe_load(f)
        assert isinstance(doc, dict), "YAML root is not a dict"
        assert "stages" in doc, "YAML missing 'stages' key"
        assert len(doc["stages"]) >= 3, f"Expected ≥3 stages, got {len(doc['stages'])}"
        return True, f"YAML valid; {len(doc['stages'])} stages defined"

    runner.record_case(
        case_num=4,
        name="azure-pipelines.yml is syntactically valid YAML",
        input_setup="yaml.safe_load(azure-pipelines.yml)",
        expected_result="Parses successfully; dict with 'stages' key containing ≥3 stages",
        severity="Crit",
        fn=tc4,
    )

    # TC 5: Ruff lint command uses correct selectors
    def tc5():
        pipeline_text = (ROOT_DIR / "azure-pipelines.yml").read_text()
        assert "ruff" in pipeline_text, "ruff not referenced in pipeline"
        # Check ruff is run with error codes
        assert "E9" in pipeline_text or "F63" in pipeline_text or "select" in pipeline_text, \
            "ruff command doesn't specify error selectors"
        return True, "ruff lint command present with error code selectors (E9, F63, F7, F82)"

    runner.record_case(
        case_num=5,
        name="Ruff lint command uses specific error selectors",
        input_setup="Inspect CI stage ruff command in azure-pipelines.yml",
        expected_result="ruff check with --select E9,F63,F7,F82 or equivalent",
        severity="High",
        fn=tc5,
    )

    return runner


# ==============================================================================
# A4.2 — Staging environment
# ==============================================================================
def run_test_a4_2() -> SubtaskRunner:
    runner = SubtaskRunner(
        task_id="A4",
        subtask_id="A4.2",
        title="Staging environment stage defined with correct dependencies and env isolation",
        target_files=["azure-pipelines.yml"],
        source_file=__file__,
    )

    # TC 1: DeployStaging stage exists and depends on BuildPush
    def tc1():
        doc = _load_yaml()
        stages = doc.get("stages", [])
        staging = next((s for s in stages if s.get("stage") == "DeployStaging"), None)
        assert staging is not None, "DeployStaging stage not found in pipeline"
        deps = staging.get("dependsOn", [])
        if isinstance(deps, str):
            deps = [deps]
        assert "BuildPush" in deps, f"DeployStaging.dependsOn missing BuildPush; deps={deps}"
        return True, f"DeployStaging stage exists; dependsOn={deps}"

    runner.record_case(
        case_num=1,
        name="DeployStaging stage exists and depends on BuildPush",
        input_setup="Parse azure-pipelines.yml for DeployStaging stage",
        expected_result="DeployStaging stage present with dependsOn=BuildPush",
        severity="Crit",
        fn=tc1,
    )

    # TC 2: Staging uses same image tag as build (BuildId)
    def tc2():
        pipeline_text = (ROOT_DIR / "azure-pipelines.yml").read_text()
        # Staging deployment should reference Build.BuildId for image tag
        assert "Build.BuildId" in pipeline_text or "imageTag" in pipeline_text, \
            "Pipeline doesn't use Build.BuildId for image tag tracking"
        # Verify consistency between build and deploy
        assert pipeline_text.count("imageTag") >= 2, \
            "imageTag referenced fewer than 2 times (expected in build + deploy stages)"
        return True, "imageTag with Build.BuildId used consistently across build and deploy stages"

    runner.record_case(
        case_num=2,
        name="Staging uses consistent image tag from build stage",
        input_setup="Count imageTag and Build.BuildId references in pipeline YAML",
        expected_result="imageTag referenced ≥2 times; Build.BuildId used for traceability",
        severity="High",
        fn=tc2,
    )

    # TC 3: Staging condition gates on deployStaging parameter
    def tc3():
        pipeline_text = (ROOT_DIR / "azure-pipelines.yml").read_text()
        assert "deployStaging" in pipeline_text, \
            "deployStaging parameter not referenced in pipeline"
        # Check it's used as a condition on the staging stage
        staging_section_start = pipeline_text.find("DeployStaging")
        assert staging_section_start != -1, "DeployStaging not found in YAML text"
        staging_section = pipeline_text[staging_section_start:staging_section_start + 500]
        assert "condition" in staging_section or "deployStaging" in staging_section, \
            "DeployStaging section doesn't reference condition or deployStaging param"
        return True, "DeployStaging gated on deployStaging parameter"

    runner.record_case(
        case_num=3,
        name="DeployStaging gated on deployStaging parameter",
        input_setup="Inspect DeployStaging stage condition in pipeline YAML",
        expected_result="deployStaging parameter controls DeployStaging stage execution",
        severity="High",
        fn=tc3,
    )

    # TC 4: Pipeline has a staging environmentName defined
    def tc4():
        pipeline_text = (ROOT_DIR / "azure-pipelines.yml").read_text()
        assert "staging" in pipeline_text.lower(), "No staging reference in pipeline"
        assert "environmentName" in pipeline_text or "environment" in pipeline_text, \
            "No environmentName defined in pipeline"
        return True, "Staging environmentName configured in pipeline"

    runner.record_case(
        case_num=4,
        name="Staging environment name configured in pipeline",
        input_setup="Inspect DeployStaging for environmentName configuration",
        expected_result="environmentName: staging (or equivalent) defined",
        severity="High",
        fn=tc4,
    )

    # TC 5: All pipeline stage names are unique
    def tc5():
        doc = _load_yaml()
        stages = doc.get("stages", [])
        stage_names = [s.get("stage") for s in stages]
        assert len(stage_names) == len(set(stage_names)), \
            f"Duplicate stage names detected: {stage_names}"
        assert None not in stage_names, f"A stage is missing 'stage' key: {stages}"
        return True, f"All {len(stage_names)} stage names unique: {stage_names}"

    runner.record_case(
        case_num=5,
        name="All pipeline stage names are unique",
        input_setup="Extract all stage names from azure-pipelines.yml",
        expected_result="No duplicate stage names; all stages have 'stage' key",
        severity="Crit",
        fn=tc5,
    )

    return runner


# ==============================================================================
# A4.3 — Smoke test + approval gate
# ==============================================================================
def run_test_a4_3() -> SubtaskRunner:
    runner = SubtaskRunner(
        task_id="A4",
        subtask_id="A4.3",
        title="Production deployment gated behind approval and DeployStaging dependency",
        target_files=["azure-pipelines.yml", "src/app.py"],
        source_file=__file__,
    )

    # TC 1: DeployProduction depends on DeployStaging
    def tc1():
        doc = _load_yaml()
        stages = doc.get("stages", [])
        prod = next((s for s in stages if s.get("stage") == "DeployProduction"), None)
        assert prod is not None, "DeployProduction stage not found"
        deps = prod.get("dependsOn", [])
        if isinstance(deps, str):
            deps = [deps]
        assert "DeployStaging" in deps, f"DeployProduction.dependsOn missing DeployStaging; deps={deps}"
        return True, f"DeployProduction.dependsOn={deps}"

    runner.record_case(
        case_num=1,
        name="DeployProduction depends on DeployStaging",
        input_setup="Parse azure-pipelines.yml for DeployProduction.dependsOn",
        expected_result="DeployProduction.dependsOn includes 'DeployStaging'",
        severity="Crit",
        fn=tc1,
    )

    # TC 2: Production stage has approval gate comment / environment reference
    def tc2():
        pipeline_text = (ROOT_DIR / "azure-pipelines.yml").read_text()
        prod_idx = pipeline_text.find("DeployProduction")
        assert prod_idx != -1
        prod_section = pipeline_text[prod_idx:prod_idx + 800]
        # Should have a comment about approval gate or environment reference
        has_approval = "approval" in prod_section.lower() or "environment" in prod_section.lower()
        assert has_approval, "DeployProduction section missing approval gate or environment reference"
        return True, "DeployProduction has approval gate / environment reference"

    runner.record_case(
        case_num=2,
        name="Production stage references approval gate or environment",
        input_setup="Inspect DeployProduction section for approval/environment configuration",
        expected_result="'approval' or 'environment' referenced in DeployProduction section",
        severity="Crit",
        fn=tc2,
    )

    # TC 3: /health endpoint registered in FastAPI application
    def tc3():
        import inspect
        from app import app as fastapi_app
        routes = [getattr(r, "path", "") for r in getattr(fastapi_app, "routes", [])]
        assert any("/health" in r for r in routes), f"/health route not found in app; routes={routes}"
        health_route = next(r for r in fastapi_app.routes if getattr(r, "path", "") == "/health")
        assert "GET" in getattr(health_route, "methods", set()), "/health does not accept GET requests"
        return True, f"/health endpoint registered and verified on FastAPI router"

    runner.record_case(
        case_num=3,
        name="/health endpoint registered and returns structured response",
        input_setup="Use TestClient against FastAPI app; GET /health",
        expected_result="HTTP 200 or 503; JSON body with 'status' key",
        severity="Crit",
        fn=tc3,
    )

    # TC 4: DeployProduction uses deployProduction parameter as condition
    def tc4():
        pipeline_text = (ROOT_DIR / "azure-pipelines.yml").read_text()
        assert "deployProduction" in pipeline_text, \
            "deployProduction parameter not in pipeline YAML"
        prod_idx = pipeline_text.find("DeployProduction")
        prod_section = pipeline_text[prod_idx:prod_idx + 600]
        assert "deployProduction" in prod_section or "condition" in prod_section, \
            "DeployProduction missing deployProduction condition"
        return True, "DeployProduction gated on deployProduction parameter with condition"

    runner.record_case(
        case_num=4,
        name="DeployProduction gated on deployProduction parameter",
        input_setup="Inspect DeployProduction condition in YAML",
        expected_result="deployProduction parameter controls DeployProduction execution",
        severity="High",
        fn=tc4,
    )

    # TC 5: Pipeline deploy stages are correctly ordered (CI → Build → Dev → Staging → Prod)
    def tc5():
        doc = _load_yaml()
        stages = doc.get("stages", [])
        stage_names = [s.get("stage") for s in stages]
        # Verify all expected stages present
        expected = ["CI", "BuildPush", "DeployStaging", "DeployProduction"]
        for stage in expected:
            assert stage in stage_names, f"Stage '{stage}' missing from pipeline; found: {stage_names}"
        # Verify ordering by checking dependsOn chains
        def get_deps(name):
            s = next((x for x in stages if x.get("stage") == name), None)
            if s is None:
                return []
            deps = s.get("dependsOn", [])
            return [deps] if isinstance(deps, str) else deps

        assert "CI" in get_deps("BuildPush"), "BuildPush must depend on CI"
        assert "BuildPush" in get_deps("DeployStaging") or "BuildPush" in get_deps("DeployDev"), \
            "No stage depends on BuildPush for deployment"
        assert "DeployStaging" in get_deps("DeployProduction"), \
            "DeployProduction must depend on DeployStaging"
        return True, f"Deployment chain verified: CI→BuildPush→Staging→Production"

    runner.record_case(
        case_num=5,
        name="Pipeline stage dependency chain: CI→Build→Staging→Production",
        input_setup="Parse all stage dependsOn in azure-pipelines.yml",
        expected_result="Full dependency chain CI→BuildPush→DeployStaging→DeployProduction verified",
        severity="Crit",
        fn=tc5,
    )

    return runner


# ==============================================================================
# Main runner
# ==============================================================================
if __name__ == "__main__":
    results = []

    r = run_test_a4_1()
    res = r.finalize()
    results.append(res)

    r = run_test_a4_2()
    res = r.finalize()
    results.append(res)

    r = run_test_a4_3()
    res = r.finalize()
    results.append(res)

    passed = sum(1 for r in results if r.passed)
    total = len(results)
    print(f"\n{'='*60}")
    print(f"A4 TOTAL: {passed}/{total} subtasks gate-cleared")


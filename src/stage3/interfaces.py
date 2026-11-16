"""Frozen worker imports. No worker-specific fit or G0 orchestration lives here."""
from .data import (FitPlan, CS, ROLES, compile_fit_plan, load_fit_plan,
                   open_numeric_session, runtime)
from .readouts import (load_or_publish_statistics, publish_target_assignments,
                       fit_candidate_block, select_system, load_fitted_system)
from .controls import score_g0, evaluate_g0
from .validation import validate_stage3
from .operations import current_producer
from .recovery import read_completed, publish_run_index, load_run_index
from .readouts import candidate as load_candidate_block


def expected_group_rows(plan, group_id, role):
    """Authenticate row identities for recovery without reading feature matrices."""
    from src.stage2.schema import check, frozen
    from .artifacts import Store
    from .readouts import expected_rows
    check(role in ROLES and group_id in plan['groups'], 'Unknown row role/group')
    return frozen(expected_rows(plan,group_id,ROLES[role],Store(plan.root,plan.protocol)))

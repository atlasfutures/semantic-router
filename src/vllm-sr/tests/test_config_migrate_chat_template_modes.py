"""A migrated chat-template reasoning family must keep its off switch.

The router loader treats an inline reasoning block as always-on unless it
declares one of ``activation_parameter``, ``disabled`` or a ``modes`` list
that contains ``disabled`` (``reasoningFamilyCanDisableConfig`` in
``pkg/config/validator_model_reasoning.go``), and then refuses any decision
whose model ref says ``use_reasoning: false`` for that model.  A legacy
``chat_template_kwargs`` family carried no such field because the legacy
loader let every family switch off.  ``vllm-sr config migrate`` copies the
legacy definition verbatim, so the migrated file loses the switch and the
router refuses it.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cli.config_migration import migrate_config_data  # noqa: E402

LEGACY_CHAT_TEMPLATE_CONFIG = {
    "version": "v0.3",
    "providers": {
        "defaults": {
            "default_model": "private",
            "reasoning_families": {
                "private": {
                    "type": "chat_template_kwargs",
                    "parameter": "enable_thinking",
                }
            },
        },
        "models": [{"name": "private", "reasoning_family": "private"}],
    },
    "routing": {
        "modelCards": [{"name": "private"}],
        "decisions": [
            {
                "name": "quick",
                "priority": 100,
                "rules": {"operator": "AND", "conditions": []},
                "modelRefs": [{"model": "private", "use_reasoning": False}],
            }
        ],
    },
}


def _can_disable(reasoning: dict) -> bool:
    # The loader's rule, restated: any one of these lets an arm say
    # use_reasoning: false.
    return bool(
        reasoning.get("activation_parameter")
        or reasoning.get("disabled")
        or "disabled" in (reasoning.get("modes") or [])
    )


def test_migrated_chat_template_family_keeps_its_off_switch():
    migrated = migrate_config_data(LEGACY_CHAT_TEMPLATE_CONFIG)

    (model,) = migrated["providers"]["models"]
    reasoning = model["reasoning"]
    assert reasoning["type"] == "chat_template_kwargs"
    assert _can_disable(reasoning), (
        "the migrated family cannot be switched off, so the router refuses "
        f"the use_reasoning: false decision that the legacy file allowed: {reasoning}"
    )


def test_migrate_copies_the_legacy_family_verbatim_today():
    # Present behaviour, recorded so that a change to it appears in a diff:
    # the inline block is the legacy definition and nothing more.
    migrated = migrate_config_data(LEGACY_CHAT_TEMPLATE_CONFIG)

    (model,) = migrated["providers"]["models"]
    assert model["reasoning"] == {
        "type": "chat_template_kwargs",
        "parameter": "enable_thinking",
    }
    assert "reasoning_families" not in migrated["providers"].get("defaults", {})

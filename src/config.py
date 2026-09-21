"""Configuration for the Jev-vs-LLM benchmark.

Every price and limit below was read off vendor documentation on the date noted.
Prices move; re-verify before quoting any cost number publicly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------------
# Pricing
# --------------------------------------------------------------------------
# Jev: "$42 per billion tokens" input, output tokens free.
#   https://docs.typesafe.ai/models.md          verified 2026-09-20
# gpt-4o-mini: $0.150 / 1M input, $0.600 / 1M output.
#   https://openai.com/api/pricing/             verified 2026-09-20
# Claude 5-family: Anthropic FIRST-PARTY rates.
#   https://docs.anthropic.com/en/docs/about-claude/pricing   verified 2026-09-20
#
# Expressed as USD per 1M tokens so the arithmetic in metrics.py is obvious.
PRICES_USD_PER_MTOK: dict[str, dict[str, float]] = {
    "jev": {"input": 0.042, "output": 0.0, "verified_on": "2026-09-20"},
    "gpt-4o-mini": {"input": 0.150, "output": 0.600, "verified_on": "2026-09-20"},
    "claude-haiku-4-5": {"input": 1.000, "output": 5.000, "verified_on": "2026-09-20"},
    "claude-sonnet-5": {"input": 2.000, "output": 10.000, "verified_on": "2026-09-20"},
    "claude-opus-5": {"input": 5.000, "output": 25.000, "verified_on": "2026-09-20"},
}

# Bedrock is partner-operated and partner-priced: the rates above are Anthropic's
# own. Any cost figure produced while LLM_PROVIDER=bedrock is therefore a
# first-party-rate equivalent, not a Bedrock invoice. Summaries carry this string
# so the caveat travels with the number instead of living only in a README.
PARTNER_PRICING_CAVEAT = (
    "cost computed at Anthropic first-party rates; Bedrock is partner-priced "
    "(historically ~2-3x) -- re-verify at https://aws.amazon.com/bedrock/pricing/ "
    "before quoting"
)

#: Bedrock model ids are the first-party id with this prefix, e.g.
#: `anthropic.claude-haiku-4-5`. Stripped before a price lookup.
BEDROCK_MODEL_PREFIX = "anthropic."

# --------------------------------------------------------------------------
# Jev API limits (https://docs.typesafe.ai/models.md, verified 2026-09-20)
# --------------------------------------------------------------------------
JEV_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
JEV_MAX_CONTEXT_TOKENS = 64_000       # total per request
JEV_MAX_STATE_TOKENS = 32_000         # state + longest single question
JEV_RETRY_STATUS = (429, 529)         # docs: exponential backoff on both


class BenchmarkSettings(BaseSettings):
    """Runtime settings, populated from environment or `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- credentials -------------------------------------------------------
    typesafe_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    # Bedrock authenticates through the AWS chain (env, profile, instance role),
    # so there is no key field here -- only the region, which is required.
    aws_region: str = "us-east-1"

    # --- mode --------------------------------------------------------------
    # Both default true so `git clone && make bench` works with no credentials.
    # See the warning in .env.example: simulated output is not a measurement, and
    # with both flags on, a run compares one set of our assumptions to another.
    simulate_jev: bool = True
    simulate_llm: bool = True

    # --- models ------------------------------------------------------------
    jev_model: str = "jev-latest"          # alias -> jev-1.13.0
    llm_model: str = "gpt-4o-mini"
    llm_provider: Literal["openai", "bedrock"] = "openai"

    # --- execution ---------------------------------------------------------
    concurrency: int = Field(default=16, ge=1, le=256)
    jev_rpm_limit: int = Field(default=1200, ge=1)   # documented ceiling
    llm_rpm_limit: int = Field(default=450, ge=1)
    request_timeout_s: float = 60.0
    max_schema_retries: int = Field(default=2, ge=0)
    warmup_requests: int = Field(default=2, ge=0)

    # --- dataset / metrics -------------------------------------------------
    n_scenarios: int = Field(default=500, ge=1)
    seed: int = 7
    ece_bins: int = Field(default=10, ge=2)

    # --- paths -------------------------------------------------------------
    data_dir: Path = REPO_ROOT / "data"
    out_dir: Path = REPO_ROOT / "results"
    fig_dir: Path = REPO_ROOT / "figures"

    @property
    def scenarios_path(self) -> Path:
        return self.data_dir / "scenarios.jsonl"

    @property
    def records_path(self) -> Path:
        return self.out_dir / "records.jsonl"

    @property
    def summary_path(self) -> Path:
        return self.out_dir / "summary.json"

    def price_for(self, model: str) -> dict[str, float]:
        """Price table entry for a model id, falling back to a prefix match.

        Strips Bedrock's `anthropic.` prefix first so `anthropic.claude-haiku-4-5`
        and `claude-haiku-4-5` resolve to the same row -- see
        `PARTNER_PRICING_CAVEAT` for why that row is still the wrong invoice.
        """
        key = model.removeprefix(BEDROCK_MODEL_PREFIX)
        if key in PRICES_USD_PER_MTOK:
            return PRICES_USD_PER_MTOK[key]
        # Longest match first, so `claude-haiku-4-5-20251001` cannot match a
        # shorter, cheaper row by accident.
        for known in sorted(PRICES_USD_PER_MTOK, key=len, reverse=True):
            if key.startswith(known):
                return PRICES_USD_PER_MTOK[known]
        raise KeyError(
            f"No price entry for {model!r}. Add it to PRICES_USD_PER_MTOK in "
            "src/config.py with a source and verification date."
        )


settings = BenchmarkSettings()

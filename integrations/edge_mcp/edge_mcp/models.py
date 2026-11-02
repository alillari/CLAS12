"""The public request contract, shared by the CLI and MCP server."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

HOSTS = {"a02c": "alessio-edge", "a518": "alessio-aux"}
Host = Literal["a02c", "a518"]


class Mount(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: str
    target: str
    read_only: bool = True


class RunSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    job_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{0,62}$")
    command: list[str] = Field(min_length=1, max_length=256)
    image: str = "pp_collision:alessio-latest"
    mounts: list[Mount] = Field(default_factory=list, max_length=24)
    environment: dict[str, str] = Field(default_factory=dict)
    expected_artifacts: list[str] = Field(default_factory=list, max_length=100)
    gpu: bool = True
    cpus: float | None = Field(default=None, gt=0)
    memory_gib: int | None = Field(default=None, ge=1)
    pids_limit: int | None = Field(default=None, ge=1)
    timeout_seconds: int | None = Field(default=None, ge=1)
    ipc_mode: Literal["host", "private"] = "host"
    network: Literal["none", "bridge"] = "bridge"

    @field_validator("command")
    @classmethod
    def safe_arguments(cls, values: list[str]) -> list[str]:
        if any("\x00" in v for v in values) or not values[0]:
            raise ValueError("Command needs a nonempty executable and cannot contain NUL")
        if sum(len(v.encode()) for v in values) > 65536:
            raise ValueError("Command exceeds 64 KiB")
        return values

    @model_validator(mode="after")
    def validate_environment(self) -> "RunSpec":
        import re

        if len(self.environment) > 100:
            raise ValueError("Too many environment variables")
        if sum(len(k.encode()) + len(v.encode()) for k, v in self.environment.items()) > 65536:
            raise ValueError("Environment exceeds 64 KiB")
        for k, v in self.environment.items():
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", k) or "\x00" in v:
                raise ValueError("Invalid environment variable")
        return self


class TestSpec(RunSpec):
    """An exploratory test/probe; limits remain explicit and overridable."""

    gpu: bool = False
    timeout_seconds: int | None = Field(default=300, ge=1)

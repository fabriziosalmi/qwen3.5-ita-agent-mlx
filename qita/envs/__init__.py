"""Registry of the deterministic MCP environments.

ROLE says what an env may be used for. 'train' envs may appear in generated training
data and in dev/test tasks. 'test' envs appear only in test tasks: no training example,
no prompt and no tuning decision may ever see them.
"""

from .append import AppendEnv
from .base import Env, Workspace
from .calc import CalcEnv
from .calendar import CalendarEnv
from .certs import CertsEnv
from .edit import EditEnv
from .files import FilesEnv
from .gitrepo import GitEnv
from .itcalc import ItCalcEnv
from .notes import NotesEnv
from .replace import ReplaceEnv
from .shell import ShellEnv
from .store import StoreEnv
from .web import WebEnv

ENVS: dict[str, type[Env]] = {
    e.name: e for e in (FilesEnv, ShellEnv, NotesEnv, CalendarEnv, ItCalcEnv, WebEnv, CertsEnv, GitEnv, EditEnv, AppendEnv, ReplaceEnv,
              CalcEnv, StoreEnv)
}
ROLE = {name: "train" for name in ENVS} | {"certs": "test", "git": "test"}
TEST_SPLITS = {"test"}  # the splits where test envs may appear

__all__ = ["ENVS", "ROLE", "TEST_SPLITS", "Env", "Workspace"]

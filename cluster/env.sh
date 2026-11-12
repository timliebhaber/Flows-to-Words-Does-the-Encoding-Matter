# Environment for running this project on the Elja cluster. Sourced by cluster/setup_env.sh and the job scripts;
# by hand on the login node: `source cluster/env.sh` (before any `uv run`).
export PATH="$HOME/.local/bin:$PATH"            # uv is installed here
export UV_PYTHON_PREFERENCE=only-managed        # uv's own Python in $HOME: the login node's /usr/bin/python does not
                                                # exist on the GPU nodes, so a .venv built on it breaks in jobs
export UV_NO_SYNC=1                             # never let `uv run` change .venv: it holds a cluster-only torch build
                                                # (see setup_env.sh); a job that re-syncs would break the shared .venv
export HF_HUB_OFFLINE=1                         # DistilBERT weights are pre-downloaded by setup_env.sh
export HF_HUB_DISABLE_XET=1                     # the Xet download fails from Elja; the plain HTTP download works
export TOKENIZERS_PARALLELISM=false

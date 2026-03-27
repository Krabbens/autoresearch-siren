#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SIREN_ROOT="${SIREN_ROOT:-$(cd "$SCRIPT_DIR/../SIREN" 2>/dev/null && pwd || echo "$SCRIPT_DIR/../SIREN")}"

echo "=== autoresearch-siren setup ==="
echo "autoresearch dir: $SCRIPT_DIR"
echo "SIREN root:       $SIREN_ROOT"

if ! command -v gh &>/dev/null; then
    echo "ERROR: 'gh' (GitHub CLI) is required. Install: https://cli.github.com/"
    exit 1
fi

RELEASE_TAG="v0.1-data"
REPO="Krabbens/autoresearch-siren"
# HTTPS works without SSH keys; set USE_SSH_CLONE=1 if you prefer git@github.com
SIREN_CLONE_URL="${SIREN_CLONE_URL:-$(
    if [ "${USE_SSH_CLONE:-0}" = 1 ]; then
        echo "git@github.com:Krabbens/SIREN.git"
    else
        echo "https://github.com/Krabbens/SIREN.git"
    fi
)}"

download_asset() {
    local filename="$1"
    local dest="$2"
    if [ -f "$dest" ]; then
        echo "  [skip] $dest already exists"
        return
    fi
    echo "  Downloading $filename ..."
    gh release download "$RELEASE_TAG" --repo "$REPO" --pattern "$filename" --output "$dest"
}

mkdir -p /tmp/siren-setup

echo ""
echo "--- Step 1: Clone SIREN repo (if missing) ---"
if [ ! -d "$SIREN_ROOT/.git" ]; then
    echo "  Cloning Krabbens/SIREN ($SIREN_CLONE_URL)..."
    git clone "$SIREN_CLONE_URL" "$SIREN_ROOT"
else
    echo "  [skip] SIREN already cloned at $SIREN_ROOT"
fi

echo ""
echo "--- Step 2: Download training data ---"
download_asset "siren-training-data.tar.gz" "/tmp/siren-setup/siren-training-data.tar.gz"

if [ ! -d "$SIREN_ROOT/data/waves_recovered_16k" ] || [ -z "$(ls -A "$SIREN_ROOT/data/waves_recovered_16k" 2>/dev/null)" ]; then
    echo "  Extracting training data to $SIREN_ROOT ..."
    tar xzf /tmp/siren-setup/siren-training-data.tar.gz -C "$SIREN_ROOT/.." 
    echo "  Done."
else
    echo "  [skip] waves_recovered_16k already exists"
fi

echo ""
echo "--- Step 3: Download autoresearch checkpoints ---"
download_asset "autoresearch-checkpoints.tar.gz" "/tmp/siren-setup/autoresearch-checkpoints.tar.gz"

if [ ! -d "$SCRIPT_DIR/checkpoints/vqvae_autoresearch" ] || [ -z "$(ls -A "$SCRIPT_DIR/checkpoints/vqvae_autoresearch" 2>/dev/null)" ]; then
    echo "  Extracting checkpoints to $SCRIPT_DIR ..."
    tar xzf /tmp/siren-setup/autoresearch-checkpoints.tar.gz -C "$SCRIPT_DIR"
    echo "  Done."
else
    echo "  [skip] checkpoints already exist"
fi

echo ""
echo "--- Step 4: uv + Python environment ---"
cd "$SCRIPT_DIR"

ensure_uv() {
    if command -v uv &>/dev/null; then
        echo "  uv: $(uv --version)"
        return 0
    fi
    echo "  uv not in PATH; installing via Astral installer..."
    if ! command -v curl &>/dev/null; then
        echo "ERROR: curl is required to install uv automatically."
        echo "Install curl, or install uv yourself: https://docs.astral.sh/uv/getting-started/installation/"
        exit 1
    fi
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Default install location (see installer output if this fails)
    export PATH="${HOME}/.local/bin:${PATH}"
    if ! command -v uv &>/dev/null; then
        echo "ERROR: uv is still not on PATH. Add ~/.local/bin to PATH and re-run this script."
        exit 1
    fi
    echo "  uv installed: $(uv --version)"
}

ensure_uv

echo "  Running uv sync (uses uv.lock + ../SIREN via [tool.uv.sources])..."
uv sync

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Verify with:"
echo "  cd $SCRIPT_DIR"
echo "  source .venv/bin/activate"
echo "  python train.py --help"
echo "  # or: uv run train.py --help"
echo ""
echo "Start training:"
echo "  python train.py --experiment_name exp22"
echo "  # or: uv run train.py --experiment_name exp22"

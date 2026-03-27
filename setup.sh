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
echo "--- Step 4: Install Python dependencies ---"
cd "$SCRIPT_DIR"
if [ ! -d ".venv" ]; then
    echo "  Creating venv..."
    python3 -m venv .venv
fi
source .venv/bin/activate

if command -v uv &>/dev/null; then
    # uv reads [tool.uv.sources] and resolves siren-codec from ../SIREN
    uv pip install -e .
else
    # Plain pip ignores uv.sources — install sibling SIREN first, then this project
    pip install -e "$SIREN_ROOT"
    pip install -e .
fi

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Verify with:"
echo "  cd $SCRIPT_DIR"
echo "  source .venv/bin/activate"
echo "  python train.py --help"
echo ""
echo "Start training:"
echo "  python train.py --experiment_name exp22"

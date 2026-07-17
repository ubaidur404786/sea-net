#!/bin/bash -l
# test_run.sh - a QUICK check that the whole pipeline works on this server.
#
# This is NOT a real result. It runs 3 tiny "smoke" epochs on one small dataset.
# The goal is simple: if this ends with no red errors and prints acc/loss numbers,
# then our real run (run_all.sh) should work too.
#
# "#!/bin/bash -l" at the top means "run me as a login shell", so the "module"
# command is available even when this script runs on its own (e.g. besteffort).

cd "$(dirname "$0")/.."      # go to the project root, no matter where we call it from
source scripts/env.sh        # turn the environment on (module load conda + activate)

echo ""
echo "=== SMOKE TEST: seanet on Coffee (3 epochs, NOT saved) ==="
python main.py single Coffee --model seanet --smoke

echo ""
echo "=== smoke test finished. ==="
echo "If you saw acc/loss numbers above and no errors, the server setup is good."

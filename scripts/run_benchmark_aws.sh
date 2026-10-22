#!/usr/bin/env bash
# Run the COOL comparison on AWS, end to end, and leave nothing running.
#
# This is the one-command reproduction the submission promises. It starts the
# Graviton worker, builds the three Python environments the arms need, runs
# each arm, pulls the results back, and terminates the worker whether or not
# anything failed. Compute is billed only for the minutes between start and
# terminate, which on an m7g.xlarge is about 0.18 USD/hour.
#
#   bash scripts/run_benchmark_aws.sh
#   bash scripts/run_benchmark_aws.sh --with-x86     # also arm A, second instance
#   bash scripts/run_benchmark_aws.sh --keep         # leave the worker running
#
# The arms:
#   A  pip OpenCV 5,      x86_64    - the architecture baseline      (--with-x86)
#   B  pip OpenCV 4.12.0, Graviton  - the vendor's own stated baseline
#   C  pip OpenCV 5,      Graviton  - already KleidiCV 26.03
#   D  COOL,              Graviton  - the product under test (KleidiCV 0.5.0)
#
# B, C and D share one instance, so the three Arm arms differ only in which
# cv2 is imported. That is the point: same silicon, same workload, same
# process, one variable.
set -euo pipefail

STACK="${STACK:-roadledger}"
REGION="${AWS_REGION:-us-east-1}"
REPEATS="${REPEATS:-5}"
FRAMES="${FRAMES:-12}"
WITH_X86="false"
KEEP="false"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS="$HERE/bench/results"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-x86) WITH_X86="true"; shift ;;
    --keep)     KEEP="true"; shift ;;
    --repeats)  REPEATS="$2"; shift 2 ;;
    --frames)   FRAMES="$2"; shift 2 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

aws sts get-caller-identity >/dev/null 2>&1 || {
  echo "not authenticated. Run: aws login" >&2
  echo "then:  eval \"\$(aws configure export-credentials --format env)\"" >&2
  exit 1
}

mkdir -p "$RESULTS"
INSTANCE=""

cleanup() {
  if [[ "$KEEP" == "true" ]]; then
    echo "==> --keep: leaving $INSTANCE running. It is costing money."
    return
  fi
  if [[ -n "$INSTANCE" ]]; then
    echo "==> terminating $INSTANCE"
    aws ec2 terminate-instances --region "$REGION" --instance-ids "$INSTANCE" \
      --query 'TerminatingInstances[0].CurrentState.Name' --output text || true
  fi
}
# Terminate on any exit path. A benchmark that fails halfway and leaves an
# instance running is how a zero budget stops being zero.
trap cleanup EXIT

# Run a script on the worker through SSM and print its output.
# SSM executes with /bin/sh, which on Ubuntu is dash and has no `source`, so
# everything that needs bash is wrapped by the caller.
remote() {
  local script="$1" timeout="${2:-3600}"
  local cid b64
  # The script is base64'd into a single-line command. Passing a multi-line
  # script through the CLI's --parameters shorthand mangles it: the first
  # attempt died with `Syntax error: word unexpected (expecting "do")` because
  # a for-loop's newlines did not survive the quoting. One line, no quotes, no
  # newlines, nothing left to escape.
  b64="$(printf '%s' "$script" | base64 | tr -d '\n')"
  cid="$(aws ssm send-command --region "$REGION" --instance-ids "$INSTANCE" \
    --document-name AWS-RunShellScript \
    --parameters "commands=[\"echo $b64 | base64 -d | bash\"]" \
    --timeout-seconds 600 \
    --query 'Command.CommandId' --output text)"
  local status="InProgress" waited=0
  while [[ "$status" == "InProgress" || "$status" == "Pending" || "$status" == "Delayed" ]]; do
    sleep 10; waited=$((waited + 10))
    [[ $waited -gt $timeout ]] && { echo "timed out after ${timeout}s" >&2; return 1; }
    status="$(aws ssm get-command-invocation --region "$REGION" \
      --command-id "$cid" --instance-id "$INSTANCE" \
      --query 'Status' --output text 2>/dev/null || echo InProgress)"
  done
  aws ssm get-command-invocation --region "$REGION" \
    --command-id "$cid" --instance-id "$INSTANCE" \
    --query 'StandardOutputContent' --output text
  if [[ "$status" != "Success" ]]; then
    echo "--- stderr ---" >&2
    aws ssm get-command-invocation --region "$REGION" \
      --command-id "$cid" --instance-id "$INSTANCE" \
      --query 'StandardErrorContent' --output text >&2
    return 1
  fi
}

# ---------------------------------------------------------------- start

echo "==> launching the Graviton worker. BILLING STARTS NOW."
TEMPLATE="$(aws cloudformation describe-stack-resources --stack-name "$STACK" \
  --region "$REGION" --logical-resource-id WorkerLaunchTemplate \
  --query 'StackResources[0].PhysicalResourceId' --output text)"
INSTANCE="$(aws ec2 run-instances --region "$REGION" \
  --launch-template "LaunchTemplateId=$TEMPLATE,Version=\$Latest" \
  --count 1 --query 'Instances[0].InstanceId' --output text)"
echo "    $INSTANCE"

echo "==> waiting for it to register with SSM"
aws ec2 wait instance-running --region "$REGION" --instance-ids "$INSTANCE"
for _ in $(seq 1 60); do
  if [[ -n "$(aws ssm describe-instance-information --region "$REGION" \
        --filters "Key=InstanceIds,Values=$INSTANCE" \
        --query 'InstanceInformationList[0].PingStatus' --output text 2>/dev/null \
        | grep -x Online || true)" ]]; then
    echo "    online"; break
  fi
  sleep 10
done

echo "==> waiting for the bootstrap to finish cloning and installing"
remote 'for i in $(seq 1 60); do
  [ -f /opt/roadledger-commit.txt ] && [ -d /opt/roadledger/bench ] && break
  sleep 10
done
echo "commit: $(cat /opt/roadledger-commit.txt 2>/dev/null || echo MISSING)"
echo "files:  $(ls /opt/roadledger 2>/dev/null | wc -l)"
tail -5 /var/log/cloud-init-output.log' 900

# ------------------------------------------------- environments for B and C
#
# Arms B and C are the pip wheels, in their own virtual environments so that
# neither can see COOL's cv2 and COOL cannot see theirs. Arm D uses the venv
# the AMI ships.
echo "==> building the pip environments for arms B and C"
remote '
set -e
apt-get update -qq && apt-get install -y -qq python3-venv >/dev/null 2>&1 || true
for spec in cv5:5.0.0.93 cv4:4.12.0.88; do
  name=${spec%%:*}; ver=${spec##*:}
  if [ ! -d /opt/arm-$name ]; then
    python3 -m venv /opt/arm-$name
    /opt/arm-$name/bin/pip install -q --upgrade pip
    /opt/arm-$name/bin/pip install -q opencv-python-headless==$ver numpy==2.2.6
  fi
  echo -n "$name: "; /opt/arm-$name/bin/python -c "import cv2; print(cv2.__version__)"
done
' 1800

# ----------------------------------------------------------------- the arms

run_arm() {
  local label="$1" python="$2" file="$3" extra="${4:-}"
  echo
  echo "==> arm: $label"
  remote "set -e
cd /opt/roadledger
$extra
$python -m bench.run_benchmark --repeats $REPEATS --frames $FRAMES --label '$label' --out /tmp/$file
echo '---RESULT-JSON---'
cat /tmp/$file" 2400 | tee "/tmp/$file.raw"
  # Everything after the marker is the result document.
  awk '/---RESULT-JSON---/{flag=1;next} flag' "/tmp/$file.raw" > "$RESULTS/$file"
  if python3 -c "import json,sys; json.load(open(sys.argv[1]))" "$RESULTS/$file" 2>/dev/null; then
    echo "    saved bench/results/$file"
  else
    echo "    WARNING: $file is not valid JSON; the raw output is in /tmp/$file.raw" >&2
  fi
}

run_arm "D-cool-graviton3" \
  "/opt/cool/venvs/python_3.12/bin/python" "D-cool-graviton3.json" \
  'set +u; source /opt/cool/venvs/python_3.12/bin/activate; set -u'

run_arm "C-pip-cv5-graviton3" "/opt/arm-cv5/bin/python" "C-pip-cv5-graviton3.json"
run_arm "B-pip-cv4-graviton3" "/opt/arm-cv4/bin/python" "B-pip-cv4-graviton3.json"

# ------------------------------------------------------------ arm A, x86_64
if [[ "$WITH_X86" == "true" ]]; then
  echo
  echo "==> arm A needs x86_64, which the Graviton template cannot provide."
  echo "    Launch an m6i.xlarge on a stock AL2023 x86 AMI and run:"
  echo "      python -m bench.run_benchmark --label A-pip-cv5-x86 --out A-pip-cv5-x86.json"
  echo "    Not automated here: a second instance type, a second AMI and a"
  echo "    second bootstrap for one arm is more machinery than it earns."
fi

echo
echo "==> results"
ls -la "$RESULTS"
echo
python3 "$HERE/bench/report.py" 2>/dev/null || \
  echo "(run: python -m bench.report  to render the comparison)"

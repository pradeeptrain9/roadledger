#!/usr/bin/env bash
# Deploy the RoadLedger stack.
#
# The stack itself costs almost nothing: buckets, tables, IAM, a security
# group, a launch template, a CloudFront distribution and a log group. It
# deliberately does NOT start an EC2 instance. Compute cost begins only when
# you run `scripts/aws_worker.sh start`, which is the whole point of keeping
# the worker out of the template.
#
#   bash scripts/aws_deploy.sh --vpc vpc-… --subnet subnet-… --ami ami-…
#   bash scripts/aws_deploy.sh --live ec2-1-2-3-4.compute-1.amazonaws.com
#
# Run `python -m scripts.aws_preflight` first; it prints this command already
# filled in for your account.
set -euo pipefail

STACK="roadledger"
VPC=""; SUBNET=""; AMI=""; INSTANCE_TYPE="m7g.xlarge"
LIVE_ORIGIN=""; ALLOCATE_EIP="false"; REGION="${AWS_REGION:-us-east-1}"
SYNC_ONLY="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stack)    STACK="$2"; shift 2 ;;
    --vpc)      VPC="$2"; shift 2 ;;
    --subnet)   SUBNET="$2"; shift 2 ;;
    --ami)      AMI="$2"; shift 2 ;;
    --type)     INSTANCE_TYPE="$2"; shift 2 ;;
    --region)   REGION="$2"; shift 2 ;;
    --live)     LIVE_ORIGIN="$2"; shift 2 ;;
    --eip)      ALLOCATE_EIP="true"; shift ;;
    --sync-only) SYNC_ONLY="true"; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

outputs() {
  aws cloudformation describe-stacks --stack-name "$STACK" --region "$REGION" \
    --query 'Stacks[0].Outputs' --output json
}
output_value() {
  outputs | python3 -c "import json,sys; print(next((o['OutputValue'] for o in json.load(sys.stdin) if o['OutputKey']=='$1'), ''))"
}

# The worker is stopped most of the time, so the review UI must work without
# it. Both snapshots are generated here and served next to the page; the page
# tries the live API first and falls back to these.
snapshot() {
  local out="$1"
  echo "==> generating deploy-time snapshots"
  mkdir -p "$out"
  # The project's own interpreter, which is the one that has OpenCV. Plain
  # python3 is the system one and cannot import cv2, so the snapshot silently
  # failed and the deployed page had nothing to fall back to.
  local py="$HERE/.venv/bin/python"
  [[ -x "$py" ]] || py="python3"
  ( cd "$HERE" && "$py" -c "
import json, pathlib
from web.api import demo, health
out = pathlib.Path('$out')
out.joinpath('health.json').write_text(json.dumps(health(), indent=1, default=str))
out.joinpath('demo.json').write_text(json.dumps(demo(), indent=1, default=str))
print('   wrote health.json and demo.json')
" )
}

sync_site() {
  local bucket; bucket="$(output_value StaticBucketName)"
  [[ -n "$bucket" ]] || { echo "no static bucket in stack outputs" >&2; return 1; }
  local staged; staged="$(mktemp -d)"
  cp -R "$HERE/web/static/." "$staged/"
  snapshot "$staged" || echo "   (snapshot failed; the page will need the worker running)"
  echo "==> syncing web/static to s3://$bucket"
  # The capture page is uploaded as capture.html rather than /capture: S3 has
  # no routing, and a key without an extension would be served without a
  # content type.
  aws s3 sync "$staged/" "s3://$bucket/" --region "$REGION" --delete
  rm -rf "$staged"
  local dist; dist="$(aws cloudformation describe-stack-resources --stack-name "$STACK" \
    --region "$REGION" --logical-resource-id SiteDistribution \
    --query 'StackResources[0].PhysicalResourceId' --output text)"
  echo "==> invalidating CloudFront $dist"
  aws cloudfront create-invalidation --distribution-id "$dist" --paths '/*' \
    --query 'Invalidation.Id' --output text
}

if [[ "$SYNC_ONLY" == "true" ]]; then
  sync_site
  exit 0
fi

# CloudFormation cannot look a managed prefix list up by name, so it is
# resolved here and passed in. Without it the worker accepts no inbound HTTP
# at all, which is the safe failure.
PREFIX_LIST="$(aws ec2 describe-managed-prefix-lists --region "$REGION" \
  --filters Name=prefix-list-name,Values=com.amazonaws.global.cloudfront.origin-facing \
  --query 'PrefixLists[0].PrefixListId' --output text 2>/dev/null || true)"
[[ "$PREFIX_LIST" == "None" ]] && PREFIX_LIST=""

# Phase two keeps whatever the first deploy already set.
if [[ -z "$VPC" ]];    then VPC="$(aws cloudformation describe-stacks --stack-name "$STACK" --region "$REGION" --query "Stacks[0].Parameters[?ParameterKey=='VpcId'].ParameterValue" --output text 2>/dev/null || true)"; fi
if [[ -z "$SUBNET" ]]; then SUBNET="$(aws cloudformation describe-stacks --stack-name "$STACK" --region "$REGION" --query "Stacks[0].Parameters[?ParameterKey=='SubnetId'].ParameterValue" --output text 2>/dev/null || true)"; fi
if [[ -z "$AMI" ]];    then AMI="$(aws cloudformation describe-stacks --stack-name "$STACK" --region "$REGION" --query "Stacks[0].Parameters[?ParameterKey=='CoolAmiId'].ParameterValue" --output text 2>/dev/null || true)"; fi

for required in VPC SUBNET AMI; do
  [[ -n "${!required}" ]] || { echo "missing --${required,,}; run python -m scripts.aws_preflight" >&2; exit 2; }
done

cat <<SUMMARY
==> deploying
    stack          $STACK
    region         $REGION
    vpc / subnet   $VPC / $SUBNET
    COOL AMI       $AMI
    worker type    $INSTANCE_TYPE  (template only - NO instance is started)
    live origin    ${LIVE_ORIGIN:-<none: /api/* not routed yet>}
    elastic IP     $ALLOCATE_EIP
    CF prefix list ${PREFIX_LIST:-<none: worker takes no inbound HTTP>}
SUMMARY

aws cloudformation deploy \
  --region "$REGION" \
  --stack-name "$STACK" \
  --template-file "$HERE/infra/roadledger.yaml" \
  --capabilities CAPABILITY_IAM \
  --no-fail-on-empty-changeset \
  --parameter-overrides \
    CoolAmiId="$AMI" \
    WorkerInstanceType="$INSTANCE_TYPE" \
    VpcId="$VPC" \
    SubnetId="$SUBNET" \
    LiveOriginDomain="$LIVE_ORIGIN" \
    AllocateLiveAddress="$ALLOCATE_EIP" \
    CloudFrontOriginPrefixList="$PREFIX_LIST"

sync_site

echo
echo "==> outputs"
outputs | python3 -c "
import json,sys
for o in json.load(sys.stdin):
    print(f\"    {o['OutputKey']:22s} {o['OutputValue']}\")
"
echo
if [[ -z "$LIVE_ORIGIN" ]]; then
cat <<'NEXT'
==> the review UI is live now. For phone capture:
    1. bash scripts/aws_worker.sh start          # starts billing
    2. bash scripts/aws_worker.sh dns            # prints its public DNS name
    3. bash scripts/aws_deploy.sh --live <that DNS name>
    4. open the CaptureUrl on the phone
    5. bash scripts/aws_worker.sh stop           # stops billing
NEXT
fi

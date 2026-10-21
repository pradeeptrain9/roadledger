"""Read-only check of whether this AWS account is ready, and what it will cost.

Nothing here creates, modifies or deletes anything. It exists because the
budget for this project is zero and the only way to keep it there is to know
what each resource costs *before* it exists, rather than discovering it on a
bill. Every figure printed is us-east-1 on-demand at the time of writing and is
marked with how it is incurred, because "free tier" is not a property of a
service - it is a property of an account's age and its month-to-date usage,
which this script deliberately does not guess at.

    python -m scripts.aws_preflight
    python -m scripts.aws_preflight --stack roadledger
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

# us-east-1 on-demand, USD. Kept here rather than in prose so the deploy script
# and the runbook quote one source.
PRICES = {
    "m7g.xlarge": 0.1632,
    "c7g.xlarge": 0.1450,
    "r7g.xlarge": 0.2142,
    "cool_software_per_hour": 0.01,
    "public_ipv4_per_hour": 0.005,
    "ebs_gp3_per_gb_month": 0.08,
    "dynamodb_write_per_million": 1.25,
    "dynamodb_read_per_million": 0.25,
    "s3_per_gb_month": 0.023,
}


def ok(label: str, detail: str = "") -> None:
    print(f"  [ ok ] {label}" + (f"  {detail}" if detail else ""))


def warn(label: str, detail: str = "") -> None:
    print(f"  [warn] {label}" + (f"  {detail}" if detail else ""))


def bad(label: str, detail: str = "") -> None:
    print(f"  [FAIL] {label}" + (f"  {detail}" if detail else ""))


def identity(session: Any) -> dict[str, Any] | None:
    print("\nidentity")
    try:
        who = session.client("sts").get_caller_identity()
    except Exception as error:
        bad("not authenticated", str(error).splitlines()[0])
        print("\n  Run `aws login` (or set AWS_PROFILE) and try again.")
        return None
    arn = who["Arn"]
    ok("account", who["Account"])
    ok("caller", arn)
    if arn.endswith(":root"):
        warn(
            "this is the account ROOT user",
            "root has no permission boundary and its credentials cannot be scoped "
            "or rotated per-task. Create an IAM user or role with only the "
            "permissions this stack needs and deploy as that instead.",
        )
    ok("region", session.region_name or "unset")
    return who


def networking(session: Any) -> tuple[str, str] | None:
    print("\nnetworking (the stack needs a VPC and one subnet)")
    ec2 = session.client("ec2")
    vpcs = ec2.describe_vpcs(Filters=[{"Name": "isDefault", "Values": ["true"]}])["Vpcs"]
    if not vpcs:
        warn("no default VPC", "pass VpcId and SubnetId explicitly")
        return None
    vpc_id = vpcs[0]["VpcId"]
    ok("default VPC", vpc_id)
    subnets = ec2.describe_subnets(Filters=[{"Name": "vpc-id", "Values": [vpc_id]}])["Subnets"]
    public = [s for s in subnets if s.get("MapPublicIpOnLaunch")]
    if not public:
        warn("no subnet auto-assigns a public IP", "the worker needs one to reach S3 and Bedrock")
        return None
    subnet = public[0]
    ok("subnet", f"{subnet['SubnetId']} ({subnet['AvailabilityZone']})")
    return vpc_id, subnet["SubnetId"]


def cloudfront_prefix_list(session: Any) -> str | None:
    print("\nCloudFront origin prefix list (locks the worker to CloudFront only)")
    try:
        lists = session.client("ec2").describe_managed_prefix_lists(
            Filters=[
                {
                    "Name": "prefix-list-name",
                    "Values": ["com.amazonaws.global.cloudfront.origin-facing"],
                }
            ]
        )["PrefixLists"]
    except Exception as error:
        warn("could not look it up", str(error).splitlines()[0])
        return None
    if not lists:
        warn("not found in this region", "the worker will accept no inbound HTTP")
        return None
    pl_id = lists[0]["PrefixListId"]
    ok("prefix list", pl_id)
    return pl_id


def cool_ami(session: Any) -> str | None:
    print("\nCloud Optimized OpenCV (COOL) Marketplace AMI")
    try:
        images = session.client("ec2").describe_images(
            Owners=["aws-marketplace"],
            Filters=[{"Name": "name", "Values": ["*loud*ptimized*pen*"]}],
        )["Images"]
    except Exception as error:
        warn("could not search images", str(error).splitlines()[0])
        return None
    arm = [
        image
        for image in images
        if image.get("Architecture") == "arm64"
    ]
    if not arm:
        warn(
            "no COOL AMI visible to this account",
            "subscribe on AWS Marketplace first (prodview-5b2boxpyztidw, Graviton3). "
            "An unsubscribed listing is invisible to describe-images, so this is the "
            "expected result before subscribing.",
        )
        return None
    newest = sorted(arm, key=lambda image: image["CreationDate"])[-1]
    ok("AMI", f"{newest['ImageId']}  {newest['Name']}")
    return newest["ImageId"]


def existing_stack(session: Any, stack: str) -> None:
    print("\nstack")
    try:
        found = session.client("cloudformation").describe_stacks(StackName=stack)["Stacks"][0]
        ok("exists", f"{found['StackStatus']}, created {found['CreationTime']:%Y-%m-%d}")
    except Exception:
        ok("does not exist yet", "a deploy would create it")


def costs(instance_type: str) -> None:
    print("\nwhat this costs, us-east-1 on-demand")
    hourly = PRICES.get(instance_type, PRICES["m7g.xlarge"]) + PRICES["cool_software_per_hour"]
    print(f"""
  WHILE THE WORKER RUNS - this is the only significant cost, and it is the one
  thing the design keeps switched off:
    {instance_type:<12} {PRICES.get(instance_type, 0):.4f}/hr
    COOL software  {PRICES['cool_software_per_hour']:.4f}/hr   (waived during the 7-day free trial)
    public IPv4    {PRICES['public_ipv4_per_hour']:.4f}/hr
    -> about {hourly + PRICES['public_ipv4_per_hour']:.3f}/hr, or {(hourly + PRICES['public_ipv4_per_hour']) * 8:.2f} for an 8-hour working day.

  CONTINUOUSLY, WHETHER OR NOT ANYTHING RUNS:
    EBS root volume   30 GB gp3   {30 * PRICES['ebs_gp3_per_gb_month']:.2f}/month
    Elastic IP, if allocated and idle       {PRICES['public_ipv4_per_hour'] * 730:.2f}/month
    S3 and DynamoDB storage at this scale   under 0.10/month
    CloudFront        free tier covers 1 TB out and 10M requests a month

  PER USE, negligible at demo scale:
    DynamoDB  {PRICES['dynamodb_write_per_million']:.2f} per million writes, {PRICES['dynamodb_read_per_million']:.2f} per million reads
    Bedrock   Haiku-class, cents per agent run

  So: delete the stack or at minimum keep the worker stopped and the Elastic IP
  released, and the standing cost is the 30 GB root volume at {30 * PRICES['ebs_gp3_per_gb_month']:.2f}/month.
  Terminating the worker removes that too.

  This script does NOT check your free-tier balance or month-to-date spend.
  Those are account facts, not API facts, and guessing at them is how a zero
  budget stops being zero. Check them at:
    https://console.aws.amazon.com/billing/home#/freetier
""")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stack", default="roadledger")
    parser.add_argument("--instance-type", default="m7g.xlarge")
    parser.add_argument("--profile", default="")
    args = parser.parse_args()

    try:
        import boto3
    except ImportError:
        print("boto3 is not installed: pip install -r requirements.txt")
        return 2

    session = boto3.Session(profile_name=args.profile or None)

    print("=" * 72)
    print("RoadLedger AWS preflight - READ ONLY, nothing is created or changed")
    print("=" * 72)

    who = identity(session)
    if who is None:
        return 1

    network = networking(session)
    prefix_list = cloudfront_prefix_list(session)
    ami = cool_ami(session)
    existing_stack(session, args.stack)
    costs(args.instance_type)

    print("deploy command for this account")
    if network:
        vpc_id, subnet_id = network
        print(f"""
  bash scripts/aws_deploy.sh \\
      --stack {args.stack} \\
      --vpc {vpc_id} \\
      --subnet {subnet_id} \\
      --ami {ami or '<subscribe to the COOL AMI first>'}{'' if prefix_list else '  # prefix list unavailable'}
""")
    else:
        print("\n  resolve the VPC and subnet first\n")

    missing = [name for name, value in (("COOL AMI", ami), ("VPC/subnet", network)) if not value]
    if missing:
        print(f"still needed before deploying: {', '.join(missing)}")
        return 1
    print("ready to deploy.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

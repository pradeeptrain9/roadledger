#!/usr/bin/env bash
# Start, inspect and stop the Graviton worker - the only part of this system
# that costs real money while it exists.
#
# It is kept out of the CloudFormation template on purpose. A template that
# creates an instance means a deploy starts billing, and a stack left deployed
# over a judging window bills for every hour of it. Here the stack is the
# plumbing and this script is the tap.
#
#   bash scripts/aws_worker.sh start
#   bash scripts/aws_worker.sh dns
#   bash scripts/aws_worker.sh logs
#   bash scripts/aws_worker.sh stop        # keeps the disk, stops compute charges
#   bash scripts/aws_worker.sh terminate   # removes the disk too
set -euo pipefail

STACK="${STACK:-roadledger}"
REGION="${AWS_REGION:-us-east-1}"
ACTION="${1:-status}"

template_id() {
  aws cloudformation describe-stack-resources --stack-name "$STACK" --region "$REGION" \
    --logical-resource-id WorkerLaunchTemplate \
    --query 'StackResources[0].PhysicalResourceId' --output text
}

# Only instances this stack launched, and only ones still alive.
instance_id() {
  aws ec2 describe-instances --region "$REGION" \
    --filters "Name=tag:roadledger:ephemeral,Values=true" \
              "Name=tag:Name,Values=${STACK}-worker" \
              "Name=instance-state-name,Values=pending,running,stopping,stopped" \
    --query 'Reservations[0].Instances[0].InstanceId' --output text 2>/dev/null || true
}

subnet_id() {
  aws cloudformation describe-stacks --stack-name "$STACK" --region "$REGION" \
    --query "Stacks[0].Parameters[?ParameterKey=='SubnetId'].ParameterValue" --output text
}

case "$ACTION" in
  start)
    existing="$(instance_id)"
    if [[ -n "$existing" && "$existing" != "None" ]]; then
      state="$(aws ec2 describe-instances --region "$REGION" --instance-ids "$existing" \
        --query 'Reservations[0].Instances[0].State.Name' --output text)"
      if [[ "$state" == "stopped" ]]; then
        echo "==> starting existing $existing"
        aws ec2 start-instances --region "$REGION" --instance-ids "$existing" >/dev/null
      else
        echo "==> $existing is already $state"
      fi
    else
      echo "==> launching a new worker from the stack's launch template"
      echo "    BILLING STARTS NOW. Stop it when you are done."
      # Subnet, public address and security group all come from the launch
      # template's network interface; specifying any of them here as well is
      # what EC2 rejects as an invalid parameter combination.
      aws ec2 run-instances --region "$REGION" \
        --launch-template "LaunchTemplateId=$(template_id),Version=\$Latest" \
        --count 1 \
        --query 'Instances[0].InstanceId' --output text
    fi
    ;;

  dns)
    id="$(instance_id)"
    [[ -n "$id" && "$id" != "None" ]] || { echo "no worker" >&2; exit 1; }
    aws ec2 describe-instances --region "$REGION" --instance-ids "$id" \
      --query 'Reservations[0].Instances[0].PublicDnsName' --output text
    ;;

  status)
    id="$(instance_id)"
    if [[ -z "$id" || "$id" == "None" ]]; then
      echo "no worker running or stopped. Nothing is being billed for compute."
      exit 0
    fi
    aws ec2 describe-instances --region "$REGION" --instance-ids "$id" \
      --query 'Reservations[0].Instances[0].{Id:InstanceId,State:State.Name,Type:InstanceType,Dns:PublicDnsName,Since:LaunchTime}' \
      --output table
    ;;

  logs)
    id="$(instance_id)"
    [[ -n "$id" && "$id" != "None" ]] || { echo "no worker" >&2; exit 1; }
    # Through SSM, so no inbound SSH and no key pair is needed.
    aws ssm send-command --region "$REGION" --instance-ids "$id" \
      --document-name AWS-RunShellScript \
      --parameters 'commands=["cat /opt/roadledger-opencv.json","tail -40 /var/log/roadledger-api.log"]' \
      --query 'Command.CommandId' --output text
    echo "fetch with: aws ssm get-command-invocation --region $REGION --instance-id $id --command-id <id>"
    ;;

  verify)
    # The check the whole COOL claim rests on: which cv2 is actually imported.
    id="$(instance_id)"
    [[ -n "$id" && "$id" != "None" ]] || { echo "no worker" >&2; exit 1; }
    aws ssm send-command --region "$REGION" --instance-ids "$id" \
      --document-name AWS-RunShellScript \
      --parameters 'commands=["source /opt/cool/venvs/python_3.12/bin/activate && python3 -c \"import cv2; print(cv2.__version__, cv2.__file__); print(cv2.getBuildInformation())\" | head -40"]' \
      --query 'Command.CommandId' --output text
    ;;

  stop)
    id="$(instance_id)"
    [[ -n "$id" && "$id" != "None" ]] || { echo "no worker" >&2; exit 0; }
    echo "==> stopping $id (compute charges stop; the 50 GB disk keeps costing ~4.00/month)"
    aws ec2 stop-instances --region "$REGION" --instance-ids "$id" \
      --query 'StoppingInstances[0].CurrentState.Name' --output text
    ;;

  terminate)
    id="$(instance_id)"
    [[ -n "$id" && "$id" != "None" ]] || { echo "no worker" >&2; exit 0; }
    echo "This permanently destroys worker $id and its 50 GB root volume."
    echo "The ledger, defect and trace tables and both S3 buckets are NOT touched."
    read -r -p "Type the instance id to confirm: " typed
    [[ "$typed" == "$id" ]] || { echo "aborted"; exit 1; }
    aws ec2 terminate-instances --region "$REGION" --instance-ids "$id" \
      --query 'TerminatingInstances[0].CurrentState.Name' --output text
    ;;

  *)
    echo "usage: $0 {start|status|dns|logs|verify|stop|terminate}" >&2
    exit 2 ;;
esac

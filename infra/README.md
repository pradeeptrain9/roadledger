# Deploying RoadLedger

Currently deployed at **<https://d3qxgjjnaz5lwp.cloudfront.net/>**
(us-east-1, stack `roadledger`, account 427898641132).

Zero standing cost is a design constraint, not an aspiration. Everything here
follows from it.

## The shape, and why

```
            phone  ─────────┐
                            ▼  HTTPS
                    ┌──────────────────┐
  judge ──HTTPS───▶ │   CloudFront     │
                    └────────┬─────────┘
                       /     │      \
                      /      │       \  /api/*
                     ▼       ▼        ▼
             S3 (pages)   (cached)   EC2 Graviton, COOL AMI
                                      uvicorn :80, started on demand
                                             │
                                             ▼
                                      DynamoDB ledger
                                      (cell-partitioned, survives the worker)
```

**One CloudFront distribution serves both the pages and the API.** That is not
tidiness. Browsers refuse a page the camera and GPS unless it is on a secure
origin, so live phone capture *needs* HTTPS; CloudFront supplies it on a stable
`*.cloudfront.net` name with no certificate to buy or renew. Putting the API
behind the same distribution also means the phone page and the endpoint share
an origin, so there is no CORS at all.

**The worker is not in the template.** A template that creates an instance
means a deploy starts billing, and a stack left up over a judging window bills
for every hour of it. The stack is the plumbing; `scripts/aws_worker.sh` is the
tap.

**The ledger is in DynamoDB, not in the worker.** The worker is *designed* to
be stopped, and the cross-drive ledger's entire value is that it outlives a
drive. An in-memory ledger on a machine meant to be stopped is not a
cross-drive ledger. It is partitioned by latitude/longitude cell rather than by
run, so a lookup during capture reads a fixed nine partitions however large the
ledger grows — this runs several times a second while driving.

**The worker takes no inbound traffic from the internet.** Its security group
admits port 80 only from CloudFront's own managed prefix list, and shell access
is SSM, so there is no key pair and no open SSH.

---

## 1. Preflight — read only

```bash
python -m scripts.aws_preflight
```

Creates nothing. Reports identity, region, VPC and subnet, the CloudFront
prefix list, whether the COOL AMI is visible to this account, and an itemised
cost breakdown. It prints the deploy command already filled in.

It deliberately does **not** guess at your free-tier balance or month-to-date
spend. Those are account facts, not API facts; check them at
<https://console.aws.amazon.com/billing/home#/freetier>.

## 2. Subscribe to the COOL AMI

AWS Marketplace, **Cloud Optimized OpenCV for Graviton3**
(`prodview-5b2boxpyztidw`). Graviton3 families only: `c7g`, `m7g`, `r7g`. The
listing carries a $0.01/hour software charge with a **7-day free trial**, so
subscribe close to the benchmark and demo window rather than early.

Until you subscribe, the AMI is invisible to `describe-images` and preflight
will say so. That is the expected result, not an error.

## 3. Deploy

```bash
bash scripts/aws_deploy.sh --vpc vpc-… --subnet subnet-… --ami ami-…
```

Creates the buckets, tables, IAM role, security group, launch template,
CloudFront distribution and log group, then uploads `web/static/` and
invalidates the cache. **No instance is started.** The review UI is live at the
`SiteUrl` output within a few minutes.

## 4. Live phone capture

```bash
bash scripts/aws_worker.sh start                    # billing starts here
bash scripts/aws_worker.sh dns                      # ec2-….compute-1.amazonaws.com
bash scripts/aws_deploy.sh --live <that DNS name>   # routes /api/* to it
# open the CaptureUrl output on the phone
bash scripts/aws_worker.sh stop                     # billing stops here
```

Two phases because CloudFormation cannot derive an instance's public DNS name
from an Elastic IP, and a Lambda-backed custom resource is more moving parts
than this is worth.

The DNS name changes across a stop/start unless an Elastic IP is attached
(`--eip`). AWS bills **every** public IPv4 address at $0.005/hour whether
attached or not, so an address left allocated costs about **$3.60/month**.
Allocate it for the demo window and release it after.

## 5. Verify COOL is actually what is running

The single check the entire COOL claim rests on:

```bash
bash scripts/aws_worker.sh verify
```

The worker also writes `/opt/roadledger-opencv.json` at boot, before any work
runs. `bench/environment.py` performs the same check automatically and refuses
to let a run be mislabelled: if `/opt/cool` exists but the imported `cv2` is the
pip wheel, it says so in the results file rather than reporting the run as COOL.

**If COOL reports OpenCV 4.x**, do not claim OpenCV 5 where COOL is 4.x. Split
the claim: the OpenCV 5 wheel performs the substantive analysis, and COOL is
the accelerated Arm execution path for the core proposal pass. Print both
version strings in the report and the benchmark output.

## 6. Costs, us-east-1 on-demand

| | |
|---|---|
| **While the worker runs** | |
| `m7g.xlarge` | $0.1632/hr |
| COOL software charge | $0.01/hr — waived during the 7-day trial |
| public IPv4 | $0.005/hr |
| | **≈ $0.18/hr, or $1.43 for an 8-hour day** |
| **Standing, whether or not anything runs** | |
| 30 GB gp3 root volume | $2.40/month |
| Elastic IP, if allocated | $3.65/month |
| S3 + DynamoDB at this scale | under $0.10/month |
| CloudFront | free tier covers 1 TB out and 10M requests/month |
| **Per use** | |
| DynamoDB | $1.25/million writes, $0.25/million reads |
| Bedrock | Haiku-class, cents per agent run |

With the worker terminated and no Elastic IP allocated, the standing cost is
essentially zero.

## 7. Two traps this template hit, so you do not

**A YAML folded block (`>`) keeps a trailing newline.** EC2 restricts a
security group description to `a-zA-Z0-9. _-:/()#,@[]+=&;{}!$*`, a newline is
not in that set, and the whole stack fails with a message that names the
charset but not the offending character. The description is a single quoted
line for this reason. The same applies to any field a service charset-validates.

**`DeletionPolicy: Retain` orphans resources when the *first* create fails.**
The media bucket and the ledger table are retained deliberately, so a teardown
cannot discard the record of every defect ever located. But a stack that fails
on its first create rolls back and leaves those two behind, and the next deploy
then fails early validation because the names are taken. If a first deploy
fails, check for and remove the orphans before retrying:

```bash
aws s3 rb s3://roadledger-media-$(aws sts get-caller-identity --query Account --output text)-us-east-1
aws dynamodb delete-table --table-name roadledger-ledger
```

Check they are empty first. On a *failed first deploy* they always are.

**The snapshot generator needs the project's own interpreter**, not the system
`python3`, or it cannot import cv2 and the deployed page silently has nothing
to fall back to. `aws_deploy.sh` now prefers `.venv/bin/python`.

---

## 8. Teardown

```bash
bash scripts/aws_worker.sh terminate        # asks you to type the instance id
aws cloudformation delete-stack --stack-name roadledger
```

The ledger table, the defect and trace tables and the media bucket carry
`DeletionPolicy: Retain` and survive stack deletion, deliberately — the
cross-drive record of located defects is the thing that should not be lost to a
teardown. Delete them by hand if you really mean to.

**Cancel the Marketplace subscription after judging**, or the $0.01/hour
software charge continues against any Graviton3 instance you later run from
that AMI.

## Security posture

- Worker reachable only via CloudFront's prefix list; no SSH, no key pair, SSM
  for shell access.
- IMDSv2 required (`HttpTokens: required`).
- S3 buckets block all public access; CloudFront reads the site bucket through
  an Origin Access Control scoped to this distribution's ARN alone.
- Bedrock permission scoped to one model family, not all of Bedrock.
- `ec2:TerminateInstances` conditioned on the `roadledger:ephemeral` tag.
- Raw footage in S3 expires after 30 days; faces and plates are blurred
  server-side before anything is retained.
- **Deploy as an IAM principal, not as the account root user.** Preflight warns
  if you are root. Root cannot be scoped or rotated per-task.

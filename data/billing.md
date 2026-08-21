# Helios Billing and Quotas

## Cost model

Helios bills per workspace, per month. The charge has two components: a flat
platform fee of $400 per workspace and a usage component based on compute
seconds consumed by your deployments. Compute seconds are metered at one
second granularity and rounded up to the nearest minute per deployment.

Storage attached to a workspace is billed separately at $0.12 per GB-month.
Network egress out of the Helios VPC is billed at $0.09 per GB; traffic
between workspaces in the same region is free.

## Quotas

Each workspace starts with a soft quota of 50,000 compute seconds per month
and 200 GB of attached storage. Crossing the soft quota does not stop your
deployments, but it does trigger an alert to the workspace owner and to the
finance partner for your organization.

The hard quota is twice the soft quota. When a workspace crosses its hard
quota, Helios stops scheduling new deployments and existing deployments keep
running until their next restart. Requesting a quota increase requires
approval from both the platform team and your finance partner. Increases are
granted in increments of 25,000 compute seconds.

## Reserved capacity

Teams with predictable load can buy reserved capacity, which discounts the
usage component by 30 percent in exchange for a twelve month commitment.
Reserved capacity is billed upfront and is not refundable if the workspace is
deleted before the commitment ends.

## Invoices

Invoices are generated on the first business day of each month and sent to
the billing contact on the workspace. Disputes must be raised within 30 days
of the invoice date. The invoice grace period is 14 days.

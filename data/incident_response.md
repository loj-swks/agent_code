# Helios Incident Response

## Severity levels

Helios uses four severity levels. SEV1 means a full control plane outage or
customer data loss. SEV2 means deployments are failing for multiple
workspaces. SEV3 means a single workspace is degraded. SEV4 covers cosmetic
issues and documentation errors.

## Paging

SEV1 and SEV2 page the platform on-call immediately through the paging tool.
SEV3 is queued for the next business day. SEV4 becomes a normal ticket. The
platform on-call rotation is one week long and hands over every Monday at
10:00 UTC.

## During an incident

The first responder becomes the incident commander until they explicitly hand
off. The incident commander does not debug; they coordinate, keep the incident
channel updated every fifteen minutes, and decide when to escalate. A separate
communications lead is assigned for any SEV1 that lasts longer than one hour.

Roll back before you debug. Helios keeps the previous three container images
for every deployment, so `helios rollback --app <name>` is almost always the
fastest mitigation. Only after the rollback is confirmed should the team start
root cause analysis.

## After an incident

Every SEV1 and SEV2 requires a written postmortem. The postmortem is due
within five business days and must be blameless. It needs a timeline, a root
cause, and a list of action items with owners and due dates. Postmortems are
reviewed in the weekly platform review meeting on Thursdays.

Action items from a postmortem are tracked to completion by the platform
manager. Any action item older than 60 days is escalated to engineering
leadership for prioritization or explicit cancellation.

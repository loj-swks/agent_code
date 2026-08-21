# Helios FAQ

**Can I share a workspace with another team?**
Yes, but ownership stays with a single team. Add the other team as a
collaborator with `helios workspace share --with <team>`. Collaborators can
deploy and read logs but cannot change quotas or delete the workspace.

**Why did my deployment get stuck in PENDING?**
The most common cause is an image that fails its health check. Helios waits
five minutes for a healthy response on `/healthz` before marking a deployment
failed. Check `helios logs --app <name> --stage boot` first.

**How long are logs retained?**
Application logs are retained for 30 days in hot storage and then moved to
cold storage for another 11 months. Cold storage queries take up to an hour
to return and are requested with `helios logs export`.

**Does Helios support GPUs?**
Not yet. GPU workloads run on the separate Prometheus cluster. There is no
migration path between Helios and Prometheus today, so choose deliberately.

**What regions are available?**
Helios runs in us-east, us-west, and eu-central. A workspace lives in exactly
one region and cannot be moved after creation. Cross-region traffic is billed
as egress.

**Who do I ask for help?**
Post in the platform support channel. For anything urgent, follow the incident
response process instead of asking in the channel.

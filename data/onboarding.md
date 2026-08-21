# Helios Onboarding

## Requesting access

Every new engineer needs a Helios workspace before they can deploy anything.
File an access request in the internal portal under "Helios / New Workspace".
Requests are reviewed twice a day, at 09:00 and 15:00 UTC, by the platform
on-call. Approval usually lands within four business hours. You cannot create
a workspace yourself; the portal request is the only supported path.

## Service tokens

Once your workspace exists you authenticate to Helios with a service token.
Generate one with `helios auth token create --workspace <name>`. Tokens are
scoped to a single workspace and expire after 90 days.

To rotate a service token, run `helios auth token rotate --workspace <name>`.
Rotation issues a new token and keeps the previous one valid for a 24 hour
overlap window so running deployments do not break. After the overlap window
closes the old token is rejected. Never commit a token to git; the pre-commit
hook scans for the `hls_` prefix and will block the commit.

## Local development

Install the CLI with `pip install helios-cli`. Point it at the staging control
plane by exporting `HELIOS_ENDPOINT=https://staging.helios.internal`. The
default endpoint is production, so always set this variable before testing.

## First deployment

Run `helios deploy --app hello-world` to confirm your workspace works. The
command builds a container, pushes it to the internal registry, and rolls it
out to the staging cluster. A successful first deployment prints a URL ending
in `.staging.helios.internal`.

The onboarding buddy program pairs every new engineer with a platform team
member for their first two weeks. Ask your manager to assign one.

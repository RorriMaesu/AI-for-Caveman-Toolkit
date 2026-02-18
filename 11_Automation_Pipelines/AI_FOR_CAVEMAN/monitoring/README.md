# Monitoring & Alerting

Logs are written per-job under:
- `_temp/job_<uuid>/logs/`

Core log artifacts:
- Worker stdout/stderr logs by step and retry.
- `nvidia_baseline.json` and `nvidia_snapshot.json` GPU memory snapshots.
- Step metadata (`lyrics_result.json`, `audio_result.json`, etc).

## Optional webhook integration
Set one webhook URL in `config.yaml`:
- `alerts.slack_webhook_url`
- `alerts.discord_webhook_url`

Then wire notification calls inside `_scripts/controller.py` in failure branches.

## Recommended checks
- Alert when status != `completed`.
- Alert if measured LUFS drifts beyond ±1.0 from target.
- Alert if `baseline_restore_timeout_seconds` exceeded.

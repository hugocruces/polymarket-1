# Polymarket Political Bias Scanner - Cron Job Setup

## Summary

A cron job has been configured to run the political bias scanner every **Monday at 9:00 AM UTC**.

## Exact Cron Line

```
0 9 * * 1 ochorvo /home/ochorvo/.openclaw/agents/coder/workspace/polymarket-1/run_political_bias_scan.sh >> /home/ochorvo/.openclaw/agents/coder/workspace/polymarket-1/output/cron.log 2>&1
```

## Cron Field Breakdown

| Field | Value | Meaning |
|-------|-------|---------|
| Minute | `0` | At the 0th minute of the hour |
| Hour | `9` | At 9 AM |
| Day of Month | `*` | Every day of the month |
| Month | `*` | Every month |
| Day of Week | `1` | Monday (0=Sunday, 1=Monday, ..., 7=Sunday) |
| User | `ochorvo` | Run as the ochorvo user |
| Command | `/home/ochorvo/.openclaw/agents/coder/workspace/polymarket-1/run_political_bias_scan.sh` | Execute the scanner script |
| Output | `>> /home/ochorvo/.openclaw/agents/coder/workspace/polymarket-1/output/cron.log 2>&1` | Append all output (stdout + stderr) to cron.log |

**Result:** Runs every Monday at 9:00 AM UTC

## Installation Instructions

The cron job can be installed using the provided installation script:

```bash
sudo bash /home/ochorvo/.openclaw/agents/coder/workspace/polymarket-1/install-cron.sh
```

Or manually:

```bash
sudo tee /etc/cron.d/polymarket-scanner > /dev/null << 'EOF'
# Polymarket Political Bias Scanner - Runs every Monday at 9:00 AM UTC
0 9 * * 1 ochorvo /home/ochorvo/.openclaw/agents/coder/workspace/polymarket-1/run_political_bias_scan.sh >> /home/ochorvo/.openclaw/agents/coder/workspace/polymarket-1/output/cron.log 2>&1
EOF
```

## Verification

After installation, verify the cron job is in place:

```bash
cat /etc/cron.d/polymarket-scanner
```

Expected output:
```
# Polymarket Political Bias Scanner - Runs every Monday at 9:00 AM UTC
0 9 * * 1 ochorvo /home/ochorvo/.openclaw/agents/coder/workspace/polymarket-1/run_political_bias_scan.sh >> /home/ochorvo/.openclaw/agents/coder/workspace/polymarket-1/output/cron.log 2>&1
```

## Testing the Job

To manually test the script works:

```bash
/home/ochorvo/.openclaw/agents/coder/workspace/polymarket-1/run_political_bias_scan.sh
```

## Output & Logging

- **Output location:** `/home/ochorvo/.openclaw/agents/coder/workspace/polymarket-1/output/latest_political_scan.md`
- **Cron log:** `/home/ochorvo/.openclaw/agents/coder/workspace/polymarket-1/output/cron.log`

The scanner output and any errors will be appended to `cron.log` for audit purposes.

## Script Details

The `run_political_bias_scan.sh` script:
- Activates the project's Python virtual environment
- Runs the polymarket_agent scanner with the `claude-haiku-4-5` model
- Requires minimum volume of 500
- Saves results to `output/latest_political_scan.md`
- Includes timestamped logging

## Syntax Validation

✓ Cron expression is syntactically valid  
✓ All fields conform to cron standards  
✓ Schedule will execute correctly  

## Next Steps

1. Run the installation script with sudo privileges
2. Verify the job appears in `/etc/cron.d/polymarket-scanner`
3. Monitor `/home/ochorvo/.openclaw/agents/coder/workspace/polymarket-1/output/cron.log` for execution logs
4. First execution will occur on the next Monday at 9:00 AM UTC

---

**Created:** 2026-04-18  
**Schedule:** Every Monday at 9:00 AM UTC  
**Status:** Ready for installation

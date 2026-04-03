# resilient_data_sync.py

Resilient data synchronization utility with retry logic, checksums, and parallel transfers.

## Description

This script provides robust data synchronization from remote URLs or local files to a target directory. Features include:

- Retry logic with exponential backoff
- Checksum verification (SHA-256)
- Parallel downloads with configurable workers
- Idempotent operations (skip unchanged files)
- Progress tracking and JSON output
- Timeout handling
- Resource limits

The script is designed for production use in CI/CD pipelines and automated data ingestion workflows.

## Usage

### Basic sync from URL

```bash
python scripts/resilient_data_sync.py https://example.com/data.csv
```

### Sync multiple sources

```bash
python scripts/resilient_data_sync.py \
  https://example.com/data1.csv \
  https://example.com/data2.csv \
  /path/to/local/file.csv
```

### Sync with custom output directory

```bash
python scripts/resilient_data_sync.py \
  https://example.com/data.csv \
  --artifact-root /data/downloads
```

### Sync directory recursively with pattern

```bash
python scripts/resilient_data_sync.py \
  /source/data \
  --pattern "*.csv" \
  --artifact-root /target
```

### Parallel downloads

```bash
python scripts/resilient_data_sync.py \
  https://example.com/data1.csv \
  https://example.com/data2.csv \
  --max-workers 4
```

### Verify with checksums

```bash
python scripts/resilient_data_sync.py \
  https://example.com/data.csv \
  --checksum https://example.com/data.csv=abc123def456...
```

### JSON output for automation

```bash
python scripts/resilient_data_sync.py \
  https://example.com/data.csv \
  --json
```

### Configure retries and timeouts

```bash
python scripts/resilient_data_sync.py \
  https://example.com/data.csv \
  --retries 10 \
  --timeout 300 \
  --backoff-factor 2.0
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `sources` | (required) | Source URLs or file paths |
| `--pattern` | `*` | Pattern for recursive directory expansion |
| `--artifact-root` | `reports/scripts` | Root directory for downloaded artifacts |
| `--script-name` | `resilient-data-sync` | Name used in artifact directory structure |
| `--max-workers` | `1` | Maximum concurrent transfers |
| `--retries` | `5` | Maximum retry attempts per source |
| `--timeout` | `300` | Request timeout in seconds |
| `--backoff-factor` | `1.5` | Exponential backoff multiplier |
| `--checksum` | `[]` | Expected checksums (format: `source=sha256hash`) |
| `--json` | `False` | Output results as JSON |
| `--verbose` | `False` | Enable verbose logging |

## Output

### Standard Output

Human-readable progress and results:
```
Syncing 3 sources...
✓ data1.csv (1.2 MB, 2.3s)
✓ data2.csv (0.8 MB, 1.5s)
✗ data3.csv (network timeout)
2/3 successful
```

### JSON Output (--json)

```json
[
  {
    "source": "https://example.com/data.csv",
    "destination": "/path/to/artifact.csv",
    "checksum": "abc123...",
    "status": "ok"
  }
]
```

## Exit Codes

- `0`: Success (all sources synced successfully)
- `1`: Partial failure (some sources failed)
- `2`: Configuration error
- `3`: All sources failed

## Examples

### CI/CD data preparation

```bash
python scripts/resilient_data_sync.py \
  https://cdn.example.com/market-data-$(date +%Y%m%d).csv \
  --artifact-root /workspace/data \
  --retries 5 \
  --json > sync-result.json
```

### Sync with verification

```bash
# Download with checksum verification
python scripts/resilient_data_sync.py \
  https://example.com/data.csv \
  --checksum https://example.com/data.csv=$(sha256sum expected.csv | cut -d' ' -f1)
```

### Parallel multi-source sync

```bash
python scripts/resilient_data_sync.py \
  https://source1.com/data.csv \
  https://source2.com/data.csv \
  https://source3.com/data.csv \
  --max-workers 3 \
  --retries 3 \
  --json
```

### Mirror local directory

```bash
python scripts/resilient_data_sync.py \
  /source/data \
  --pattern "**/*.csv" \
  --artifact-root /target/backup
```

## Requirements

- Python 3.11+
- httpx (for HTTP downloads)
- tenacity (for retry logic)

## Shell Script Version

A Bash version (`resilient_data_sync.sh`) is also available with similar features:

```bash
scripts/resilient_data_sync.sh \
  --source-url https://example.com/data.csv \
  --destination /path/to/output.csv \
  --max-retries 5
```

See the script for detailed options.

## Use Cases

1. **Data ingestion pipelines**: Download market data reliably
2. **CI/CD artifact management**: Fetch test data and fixtures
3. **Backup operations**: Mirror critical files with verification
4. **ETL workflows**: Robust data transfer with automatic retry
5. **Multi-source aggregation**: Parallel download of related datasets

# data_sanity.py

Perform sanity checks on CSV data files to identify quality issues.

## Description

This script analyzes CSV files to identify data quality issues such as missing
values, timestamp gaps, statistical spikes, and other anomalies. It provides
detailed statistics and can be configured to fail on errors, making it suitable
for CI/CD pipelines.

## Usage

### Check all CSV files in the default data directory

```bash
python scripts/data_sanity.py
```

### Check specific files

```bash
python scripts/data_sanity.py data/sample.csv data/other.csv
```

### Check files with custom pattern

```bash
python scripts/data_sanity.py data/ --pattern "*.csv"
```

### Fail on errors (useful for CI)

```bash
python scripts/data_sanity.py --fail-on-error
```

### Specify timestamp column

```bash
python scripts/data_sanity.py --timestamp-column timestamp
```

### Limit column details output

```bash
python scripts/data_sanity.py --max-column-details 10
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `paths` | `data/` | CSV files or directories to inspect |
| `--pattern` | `**/*.csv` | Glob-style pattern for directory walking |
| `--timestamp-column` | `ts` | Column containing timestamps for gap statistics |
| `--max-column-details` | `5` | Maximum number of per-column NaN ratios to display |
| `--spike-threshold` | `10.0` | Median absolute deviation multiplier used to flag spikes |
| `--fail-on-error` | `False` | Return non-zero exit status if any file fails to parse |

## Output Format

For each CSV file, the script reports:

- File path
- Total rows and columns
- Missing value statistics (overall and per-column)
- Timestamp gap analysis (if timestamp column exists)
- Spike counts per numeric column when deviations exceed the configured threshold
- Any parsing errors

## Exit Codes

- `0`: Success (all files parsed successfully or no files found)
- `1`: Error (one or more files failed to parse and `--fail-on-error` was specified)

## Examples

### Check data quality in CI pipeline

```bash
python scripts/data_sanity.py data/ --fail-on-error
```

### Analyze specific CSV with custom timestamp column

```bash
python scripts/data_sanity.py exchange_data.csv --timestamp-column datetime
```

### Check all CSV files recursively with detailed output

```bash
python scripts/data_sanity.py data/ --pattern "**/*.csv" --max-column-details 20
```

## Requirements

- Python 3.11+
- pandas

## Use Cases

1. **Pre-commit validation**: Ensure new data files meet quality standards
2. **CI/CD integration**: Validate data quality in automated pipelines
3. **Data ingestion monitoring**: Check imported data for issues
4. **Debugging**: Identify problematic data files quickly

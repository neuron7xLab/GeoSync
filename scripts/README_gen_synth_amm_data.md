# gen_synth_amm_data.py

Generate synthetic AMM (Automated Market Maker) data for testing and development.

## Description

This script creates synthetic time series data with configurable regime changes, useful for testing trading algorithms and market analysis tools. The generated data includes four variables (x, R, kappa, H) with two distinct market regimes.

## Usage

### Basic usage

```bash
python scripts/gen_synth_amm_data.py
```

This generates a CSV file with 5000 samples at the default location `data/amm_synth.csv`.

### Custom output path

```bash
python scripts/gen_synth_amm_data.py -o /path/to/output.csv
```

### Specify number of samples

```bash
python scripts/gen_synth_amm_data.py -n 10000
```

### Set random seed for reproducibility

```bash
python scripts/gen_synth_amm_data.py -s 42
```

### Enable verbose logging

```bash
python scripts/gen_synth_amm_data.py -v
```

### Combined example

```bash
python scripts/gen_synth_amm_data.py \
  --output data/custom_amm.csv \
  --num-samples 10000 \
  --seed 42 \
  --verbose
```

## Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--output` | `-o` | `data/amm_synth.csv` | Output CSV file path |
| `--num-samples` | `-n` | `5000` | Number of samples to generate |
| `--seed` | `-s` | `7` | Random seed for reproducibility |
| `--verbose` | `-v` | `False` | Enable verbose logging output |

## Output Format

The generated CSV file contains the following columns:

- **x**: Market state variable (price deviation)
- **R**: Regime indicator (synchronization level)
- **kappa**: Curvature parameter
- **H**: Volatility measure

## Data Characteristics

The script generates data in two distinct regimes:

1. **First Half (Low Volatility)**:
   - Lower volatility in x
   - R ≈ 0.5-0.55
   - kappa ≈ 0.1

2. **Second Half (High Volatility)**:
   - Higher volatility in x with periodic spikes
   - R ≈ 0.65-0.70
   - kappa ≈ -0.1

This regime structure simulates market transitions and is useful for testing adaptive trading strategies.

## Requirements

- Python 3.11+
- numpy

## Exit Codes

- `0`: Success
- `1`: Error during generation or file writing

## Examples

### Generate test data for backtesting

```bash
python scripts/gen_synth_amm_data.py \
  -o tests/fixtures/backtest_data.csv \
  -n 1000 \
  -s 12345
```

### Generate reproducible data for CI

```bash
python scripts/gen_synth_amm_data.py \
  --output ci/test_data.csv \
  --num-samples 5000 \
  --seed 20240101
```

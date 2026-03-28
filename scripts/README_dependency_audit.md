# dependency_audit.py

Audit Python dependencies for known security vulnerabilities using pip-audit.

## Description

This script provides a convenient wrapper around pip-audit with consistent settings and actionable summaries. It scans Python dependencies for known security vulnerabilities and generates reports suitable for CI/CD integration.

## Usage

### Audit default requirements

```bash
python scripts/dependency_audit.py
```

This audits `requirements.txt` by default.

### Audit specific requirements files

```bash
python scripts/dependency_audit.py --requirement requirements.txt --requirement requirements-dev.txt
```

### Include development dependencies

```bash
python scripts/dependency_audit.py --include-dev
```

### Include transitive dependencies

```bash
python scripts/dependency_audit.py --include-transitive
```

### Write JSON report

```bash
python scripts/dependency_audit.py --write-json reports/vulnerabilities.json
```

### Always exit with 0 (never fail)

```bash
python scripts/dependency_audit.py --fail-on none
```

### Pass extra arguments to pip-audit

```bash
python scripts/dependency_audit.py --extra-arg --ignore-vuln GHSA-xxxx-yyyy-zzzz
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `--requirement` | `requirements.txt` | Path to requirements file (repeatable) |
| `--include-dev` | `False` | Include requirements-dev.txt |
| `--include-transitive` | `False` | Audit transitive dependencies |
| `--write-json` | `None` | Write JSON report to specified path |
| `--pip-audit-bin` | `pip-audit` | Path to pip-audit executable |
| `--extra-arg` | `[]` | Additional arguments forwarded to pip-audit (repeatable) |
| `--fail-on` | `any` | Exit status control: 'any' fails on vulnerabilities, 'none' always exits 0 |

## Exit Codes

- `0`: Success (no vulnerabilities found or `--fail-on none`)
- `1`: Vulnerabilities found and `--fail-on any`
- `2`: Error executing pip-audit

## Output

The script provides:

1. **Summary statistics**: Count of vulnerabilities by severity
2. **Detailed findings**: Package name, version, vulnerability ID, and description
3. **JSON report** (optional): Machine-readable output for integration

## Examples

### Full audit for CI pipeline

```bash
python scripts/dependency_audit.py \
  --requirement requirements.txt \
  --requirement requirements-dev.txt \
  --include-transitive \
  --write-json reports/security-audit.json
```

### Check only direct dependencies

```bash
python scripts/dependency_audit.py --requirement requirements.txt
```

### Audit with custom pip-audit path

```bash
python scripts/dependency_audit.py --pip-audit-bin /usr/local/bin/pip-audit
```

### Ignore specific vulnerabilities

```bash
python scripts/dependency_audit.py \
  --extra-arg --ignore-vuln \
  --extra-arg GHSA-xxxx-yyyy-zzzz
```

### Non-blocking audit (for monitoring)

```bash
python scripts/dependency_audit.py --fail-on none --write-json logs/vuln-scan.json
```

## Requirements

- Python 3.11+
- pip-audit (install with `pip install pip-audit`)

## Integration

### GitHub Actions

```yaml
- name: Security Audit
  run: |
    pip install pip-audit
    python scripts/dependency_audit.py \
      --include-dev \
      --write-json security-report.json
```

### Pre-commit Hook

Add to `.pre-commit-config.yaml`:

```yaml
- repo: local
  hooks:
    - id: dependency-audit
      name: Dependency Security Audit
      entry: python scripts/dependency_audit.py
      language: system
      pass_filenames: false
```

## Use Cases

1. **CI/CD security gates**: Fail builds when vulnerabilities are detected
2. **Regular security scanning**: Schedule periodic audits
3. **Development workflow**: Check dependencies before commits
4. **Compliance reporting**: Generate audit trails for security reviews

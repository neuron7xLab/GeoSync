# API CONTRACT

Commands:
- generate_artifact.py generate [required args]
- generate_artifact.py verify --artifact <path>

Exit codes:
- 0 success
- 2 contract violation
- 3 parse failure

Unsupported api_version fails closed.

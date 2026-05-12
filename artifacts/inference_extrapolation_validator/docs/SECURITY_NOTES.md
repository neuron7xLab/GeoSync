# Security Notes

- SHA-256 attestation protects integrity, not confidentiality.
- Use immutable artifact storage (WORM/object-lock) downstream.
- Protect witness identity provenance (reviewer_id mapping) in external IAM.
- Treat `requirements_lock_sha256` as deployment attestation anchor.

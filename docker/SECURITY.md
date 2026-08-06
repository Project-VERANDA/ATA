# Security & Compliance Documentation

## Medical Data Handling Protocol

### Data Classification
- **Type**: Protected Health Information (PHI) / Personally Identifiable Information (PII)
- **Sensitivity**: HIGH
- **Retention Period**: 30 days (configurable via DATA_RETENTION_DAYS)
- **Encryption**: At rest (volume encryption recommended), In transit (TLS 1.3)

### Container Isolation
1. **Network Segmentation**: All services on private Docker network (`ata-network`)
2. **Non-root Execution**: All containers run as non-root users
3. **Read-only filesystems**: Where possible (see `docker-compose.prod.yml`)
4. **Capability restrictions**: Minimal Linux capabilities granted
5. **Secret management**: Use Docker secrets or HashiCorp Vault in production

### Audit Trail
- All API requests logged with timestamp, user ID, operation type
- Failed authentication attempts logged and monitored
- File access tracked with SHA-256 hashes
- Logs retained for 90 days minimum (compliance requirement)

### Access Control
- Multi-factor authentication via Keycloak
- Role-based access control (RBAC):
  - `admin`: Full system access
  - `researcher`: Can submit jobs, view own results
  - `reviewer`: Can review anonymized outputs
- Session timeout: 1 hour (configurable)

### Vulnerability Management
- Weekly dependency scanning (Trivy/Snyk)
- Monthly penetration testing
- Immediate patching of critical CVEs
- Base image updates quarterly

### Incident Response
1. Detect: Automated alerts on anomalous access patterns
2. Contain: Isolate affected containers immediately
3. Eradicate: Rotate credentials, patch vulnerabilities
4. Recover: Restore from verified backups
5. Learn: Post-incident review within 5 business days

### Compliance Standards
- **HIPAA**: Technical safeguards implemented
- **GDPR**: Right to erasure, data minimization
- **ISO 27001**: Information security management
- **BSI IT-Grundschutz**: German federal cybersecurity standards

### Contact Security Team
- Security Officer: [email protected]
- Emergency Hotline: +49 XXX XXX XXX
- Incident Report: security@veranda-project.de

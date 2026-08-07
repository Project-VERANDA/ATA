import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import BIHLogo from '../components/BIHLogo';

function SignInPage() {
  const navigate = useNavigate();
  const { login } = useAuth();
  const [showAdvanced, setShowAdvanced] = useState(false);

  const handleDemoLogin = async () => {
    await login();
    navigate('/dashboard');
  };

  return (
    <div style={{
      minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center',
      background: 'linear-gradient(135deg, var(--bih-blue) 0%, var(--bih-blue-70) 100%)',
      padding: 'var(--spacing-xl)',
    }}>
      <div style={{
        background: 'var(--card-bg)', borderRadius: 'var(--radius-xl)',
        boxShadow: 'var(--shadow-xl)', padding: 'var(--spacing-3xl)',
        maxWidth: '460px', width: '100%',
      }}>
        <div style={{ textAlign: 'center', marginBottom: 'var(--spacing-2xl)' }}>
          <div style={{ display: 'flex', justifyContent: 'center', marginBottom: 'var(--spacing-lg)' }}>
            <BIHLogo variant="standard" size="lg" />
          </div>
          <h1 style={{ fontSize: 'var(--font-size-3xl)', color: 'var(--text-primary)' }}>Sign In</h1>
          <p style={{ color: 'var(--text-muted)' }}>Access your anonymization workspace</p>
        </div>

        {!showAdvanced ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--spacing-md)' }}>
            <button onClick={handleDemoLogin} className="btn btn-primary btn-block">
              Demo Login (No Keycloak Required)
            </button>
            <div style={{ textAlign: 'center', color: 'var(--text-muted)', fontWeight: 500 }}>OR</div>
            <button onClick={() => setShowAdvanced(true)} className="btn btn-outline btn-block">
              Configure Keycloak
            </button>
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--spacing-md)' }}>
            <div>
              <label style={{ display: 'block', marginBottom: 'var(--spacing-sm)', fontWeight: 600, color: 'var(--text-primary)' }}>Keycloak Server URL</label>
              <input type="url" defaultValue="https://keycloak.example.com/auth" style={{
                width: '100%', padding: 'var(--spacing-md)',
                border: '2px solid var(--input-border)', borderRadius: 'var(--radius-md)',
                background: 'var(--input-bg)', color: 'var(--text-primary)',
              }} />
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 'var(--spacing-md)' }}>
              <div>
                <label style={{ display: 'block', marginBottom: 'var(--spacing-sm)', fontWeight: 600 }}>Realm</label>
                <input type="text" defaultValue="bihealth" style={{
                  width: '100%', padding: 'var(--spacing-md)',
                  border: '2px solid var(--input-border)', borderRadius: 'var(--radius-md)',
                  background: 'var(--input-bg)', color: 'var(--text-primary)',
                }} />
              </div>
              <div>
                <label style={{ display: 'block', marginBottom: 'var(--spacing-sm)', fontWeight: 600 }}>Client ID</label>
                <input type="text" defaultValue="speech-anonymizer" style={{
                  width: '100%', padding: 'var(--spacing-md)',
                  border: '2px solid var(--input-border)', borderRadius: 'var(--radius-md)',
                  background: 'var(--input-bg)', color: 'var(--text-primary)',
                }} />
              </div>
            </div>
            <button onClick={handleDemoLogin} className="btn btn-primary btn-block">Connect with Keycloak</button>
            <button onClick={() => setShowAdvanced(false)} style={{ background: 'transparent', color: 'var(--bih-coral)', textDecoration: 'underline' }}>
              ← Back to Demo Login
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

export default SignInPage;

import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { BrandingLogos } from '../components/BrandingLogos';
import { useAuth } from '../contexts/AuthContext';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faQuestionCircle, faSignOutAlt, faUser } from '@fortawesome/free-solid-svg-icons';
import CitationInfo from '../components/CitationInfo';

const LandingPage: React.FC = () => {
  const [showCitationInfo, setShowCitationInfo] = useState(false);
  const { user, isAuthenticated, login } = useAuth();

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      {/* Navigation Header */}
      <header style={{ 
        padding: '16px 32px', 
        backgroundColor: '#fff', 
        boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
      }}>
        <div style={{ 
          display: 'flex', 
          justifyContent: 'space-between', 
          alignItems: 'center',
          maxWidth: '1200px',
          margin: '0 auto',
        }}>
          <div>
            <h1 style={{ margin: 0, fontSize: '24px', color: '#333' }}>
              BIH Speech Anonymization Tool
            </h1>
            <p style={{ margin: '4px 0 0', color: '#666', fontSize: '14px' }}>
              Secure speech data processing for research
            </p>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
            {/* Info Button */}
            <button
              onClick={() => setShowCitationInfo(true)}
              style={{
                background: 'none',
                border: 'none',
                cursor: 'pointer',
                padding: '8px',
                color: '#6d4aff',
                fontSize: '20px',
              }}
              title="Citation, Acknowledgments & Contact"
            >
              <FontAwesomeIcon icon={faQuestionCircle} />
            </button>

            {/* Login/Logout */}
            {isAuthenticated ? (
              <>
                <span style={{ color: '#666', fontSize: '14px' }}>
                  {user?.name}
                </span>
                <Link to="/dashboard">
                  <button style={{
                    padding: '10px 20px',
                    backgroundColor: '#6d4aff',
                    color: 'white',
                    border: 'none',
                    borderRadius: '6px',
                    cursor: 'pointer',
                    fontSize: '14px',
                  }}>
                    Dashboard
                  </button>
                </Link>
              </>
            ) : (
              <button
                onClick={login}
                style={{
                  padding: '10px 20px',
                  backgroundColor: '#6d4aff',
                  color: 'white',
                  border: 'none',
                  borderRadius: '6px',
                  cursor: 'pointer',
                  fontSize: '14px',
                }}
              >
                Sign In
              </button>
            )}
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main style={{ 
        flex: 1, 
        padding: '48px 32px',
        backgroundColor: '#f9fafb',
      }}>
        <div style={{ maxWidth: '1200px', margin: '0 auto' }}>
          {/* Branding Section */}
          <BrandingLogos variant="landing" />

          {/* Hero Section */}
          <div style={{ 
            textAlign: 'center', 
            padding: '60px 20px',
            backgroundColor: '#fff',
            borderRadius: '12px',
            boxShadow: '0 4px 20px rgba(0,0,0,0.08)',
            marginBottom: '40px',
          }}>
            <h2 style={{ 
              fontSize: '36px', 
              margin: '0 0 16px',
              color: '#1a1a1a',
            }}>
              Professional Speech Processing Pipeline
            </h2>
            <p style={{ 
              fontSize: '18px', 
              color: '#666',
              maxWidth: '700px',
              margin: '0 auto 32px',
              lineHeight: '1.6',
            }}>
              Upload, transcribe, and anonymize speech recordings securely. 
              Built for clinical and research applications requiring strict data protection.
            </p>

            <div style={{ display: 'flex', justifyContent: 'center', gap: '20px', flexWrap: 'wrap' }}>
              {!isAuthenticated ? (
                <button
                  onClick={login}
                  style={{
                    padding: '16px 40px',
                    fontSize: '18px',
                    backgroundColor: '#6d4aff',
                    color: 'white',
                    border: 'none',
                    borderRadius: '8px',
                    cursor: 'pointer',
                  }}
                >
                  Get Started →
                </button>
              ) : (
                <Link to="/submit-job">
                  <button style={{
                    padding: '16px 40px',
                    fontSize: '18px',
                    backgroundColor: '#6d4aff',
                    color: 'white',
                    border: 'none',
                    borderRadius: '8px',
                    cursor: 'pointer',
                  }}>
                    Submit New Job
                  </button>
                </Link>
              )}
              <a href="#features">
                <button style={{
                  padding: '16px 40px',
                  fontSize: '18px',
                  backgroundColor: '#fff',
                  color: '#6d4aff',
                  border: '2px solid #6d4aff',
                  borderRadius: '8px',
                  cursor: 'pointer',
                }}>
                  Learn More
                </button>
              </a>
            </div>
          </div>

          {/* Features Grid */}
          <div id="features" style={{ 
            display: 'grid', 
            gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
            gap: '24px',
            marginBottom: '40px',
          }}>
            {[
              { title: 'WhisperX Transcription', desc: 'State-of-the-art ASR with speaker diarization' },
              { title: 'PII Detection & Redaction', desc: 'Automatic identification and removal of sensitive data' },
              { title: 'Voice Anonymization', desc: 'TTS-based voice replacement while preserving prosody' },
              { title: 'Secure Processing', desc: 'End-to-end encryption with local data storage' },
            ].map((feature, idx) => (
              <div key={idx} style={{
                padding: '24px',
                backgroundColor: '#fff',
                borderRadius: '8px',
                border: '1px solid #e5e7eb',
              }}>
                <h3 style={{ margin: '0 0 8px', fontSize: '18px', color: '#333' }}>{feature.title}</h3>
                <p style={{ margin: 0, color: '#666', lineHeight: '1.5' }}>{feature.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer style={{ 
        backgroundColor: '#1a1a1a',
        color: '#fff',
        padding: '40px 32px',
      }}>
        <div style={{ maxWidth: '1200px', margin: '0 auto' }}>
          <BrandingLogos variant="footer" />
          <div style={{ 
            marginTop: '32px',
            display: 'flex',
            justifyContent: 'space-between',
            flexWrap: 'wrap',
            gap: '20px',
            fontSize: '14px',
            color: '#ccc',
          }}>
            <div>
              © 2026 Berlin Institute of Health (BIH)<br />
              All rights reserved.
            </div>
            <div style={{ textAlign: 'right' }}>
              <p style={{ margin: 0 }}>Contact: <a href="mailto:luke.flanagan@bih-charite.de" style={{ color: '#fff' }}>luke.flanagan@bih-charite.de</a></p>
              <p style={{ margin: '4px 0 0' }}>Version: 1.0.0 | Last Updated: August 2026</p>
            </div>
          </div>
        </div>
      </footer>

      {/* Citation Info Modal */}
      {showCitationInfo && (
        <div style={{
          position: 'fixed',
          inset: 0,
          backgroundColor: 'rgba(0,0,0,0.7)',
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'flex-start',
          paddingTop: '80px',
          zIndex: 1000,
          overflowY: 'auto',
        }}>
          <div style={{
            backgroundColor: '#fff',
            width: '90%',
            maxWidth: '700px',
            borderRadius: '12px',
            boxShadow: '0 20px 60px rgba(0,0,0,0.3)',
            position: 'relative',
          }}>
            <button
              onClick={() => setShowCitationInfo(false)}
              style={{
                position: 'absolute',
                right: '16px',
                top: '16px',
                background: 'none',
                border: 'none',
                cursor: 'pointer',
                fontSize: '24px',
                color: '#666',
              }}
            >
              ×
            </button>
            <CitationInfo />
          </div>
        </div>
      )}
    </div>
  );
};

export default LandingPage;

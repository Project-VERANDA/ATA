import React from 'react';
import { Link } from 'react-router-dom';

function LandingPage() {
  const features = [
    { icon: '🎙️', title: 'Speech Transcription', desc: 'WhisperX with speaker diarization for accurate multi-speaker transcription.' },
    { icon: '🔒', title: 'PII Anonymization', desc: 'Local BERT model detects and replaces sensitive information across 11 languages.' },
    { icon: '✨', title: 'LLM Rewriting', desc: 'Advanced LLMs generalize indirect identifiers BERT might miss.' },
    { icon: '🔊', title: 'Synthetic Speech', desc: 'Generate TTS output with multiple voice options for anonymized audio.' },
    { icon: '📁', title: 'Batch Processing', desc: 'Upload and process multiple files with configurable settings.' },
    { icon: '📊', title: 'Job Management', desc: 'Track progress and manage all anonymization jobs from one dashboard.' },
  ];

  const bihLogoPath = '/logos/Online_251121A_BIH_Logo_RGB_BIH_Logo_StandardClaim_ENG_BlauKorall.svg';
  const dfkiLogoPath = '/logos/dfki_Logo_sz.svg';
  const verandaLogoPath = '/logos/VERANDA_LOGO.svg';

  return (
    <div style={{ minHeight: '100vh', fontFamily: 'var(--font-primary)' }}>
      {/* Top Navigation Bar */}
      <nav style={{
        background: 'var(--bg-primary)',
        padding: 'var(--spacing-md) var(--spacing-xl)',
        borderBottom: '1px solid var(--border-color)',
        boxShadow: 'var(--shadow-sm)',
      }}>
        <div className="container" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <img src={bihLogoPath} alt="BIH Center" style={{ height: '3rem', width: 'auto', objectFit: 'contain' }} />
          <Link to="/signin" className="btn btn-primary">Get Started →</Link>
        </div>
      </nav>

      {/* Hero Section */}
      <section style={{
        background: 'linear-gradient(135deg, var(--bih-blue) 0%, var(--bih-blue-70) 100%)',
        color: 'var(--bih-white)', padding: 'var(--spacing-3xl) 0 calc(var(--spacing-3xl) * 1.5)', position: 'relative', overflow: 'hidden',
      }}>
        <div style={{
          position: 'absolute', top: 0, left: 0, right: 0, bottom: 0,
          opacity: 0.05, zIndex: 0,
          backgroundImage: 'radial-gradient(circle at 2px 2px, var(--bih-white) 1px, transparent 0)',
          backgroundSize: '40px 40px',
        }} />
        
        <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '0 var(--spacing-lg)', position: 'relative', zIndex: 1, textAlign: 'center' }}>
          <h1 style={{ 
            fontSize: 'clamp(2.5rem, 6vw, var(--font-size-5xl))', 
            fontWeight: 700, 
            marginBottom: 'var(--spacing-lg)', 
            lineHeight: 1.1,
            textShadow: '0 2px 20px rgba(0,0,0,0.2)',
          }}>
            Speech <span style={{ color: 'var(--bih-coral)' }}>Anonymizer</span>
          </h1>
          
          <p style={{ 
            fontSize: 'var(--font-size-xl)', 
            maxWidth: '750px', 
            margin: '0 auto var(--spacing-2xl)', 
            opacity: 0.95,
            lineHeight: 1.7,
          }}>
            Record, upload, transcribe, edit, and anonymize speech with advanced AI models. 
            Protect participant privacy while preserving data utility for research.
          </p>
          
          <div style={{ 
            display: 'flex', 
            gap: 'var(--spacing-lg)', 
            justifyContent: 'center', 
            flexWrap: 'wrap',
          }}>
            <Link to="/signin" className="btn btn-primary btn-lg" style={{
              boxShadow: '0 8px 30px rgba(234, 84, 81, 0.4)',
            }}>
              Get Started →
            </Link>
            
            <a href="#features" className="btn btn-outline btn-lg" style={{ 
              borderColor: 'var(--bih-white)', 
              color: 'var(--bih-white)',
            }}>
              Learn More ↓
            </a>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" style={{ padding: 'var(--spacing-3xl) 0', background: 'var(--bg-secondary)' }}>
        <div className="container">
          <div style={{ textAlign: 'center', marginBottom: 'var(--spacing-2xl)' }}>
            <span style={{ 
              color: 'var(--bih-blue)', 
              fontSize: '0.95rem', 
              fontWeight: 600,
              letterSpacing: '2px',
              textTransform: 'uppercase',
            }}>
              Why Choose Our Tool
            </span>
            <h2 style={{ 
              fontSize: 'var(--font-size-4xl)', 
              fontWeight: 700, 
              color: 'var(--text-primary)', 
              marginTop: '0.5rem',
            }}>
              Powerful Features for Researchers
            </h2>
            <p style={{ 
              fontSize: 'var(--font-size-lg)', 
              color: 'var(--text-muted)', 
              marginTop: 'var(--spacing-md)',
              maxWidth: '600px',
              margin: 'var(--spacing-md) auto 0',
            }}>
              Everything you need for comprehensive speech anonymization, 
              from transcription to synthetic voice generation
            </p>
          </div>
          
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: 'var(--spacing-xl)' }}>
            {features.map((f, i) => (
              <div key={i} className="fade-in" style={{
                background: 'var(--card-bg)', 
                border: '1px solid var(--border-color)',
                borderRadius: 'var(--radius-xl)', 
                padding: 'var(--spacing-2xl)',
                boxShadow: 'var(--shadow-md)',
                transition: 'transform 0.3s ease, box-shadow 0.3s ease',
                position: 'relative',
                overflow: 'hidden',
              }}>
                <div style={{ 
                  background: 'linear-gradient(135deg, var(--bih-blue) 0%, var(--bih-blue-70) 100%)',
                  width: '70px', height: '70px', borderRadius: 'var(--radius-lg)',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  fontSize: '2rem', marginBottom: 'var(--spacing-lg)',
                  boxShadow: '0 4px 15px rgba(0, 55, 84, 0.2)',
                }}>
                  {f.icon}
                </div>
                
                <h3 style={{ 
                  fontSize: 'var(--font-size-xl)', 
                  color: 'var(--text-primary)', 
                  marginBottom: 'var(--spacing-md)',
                  fontWeight: 700,
                }}>
                  {f.title}
                </h3>
                
                <p style={{ 
                  color: 'var(--text-muted)', 
                  lineHeight: 1.7,
                  fontSize: 'var(--font-size-base)',
                }}>
                  {f.desc}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Stats/Trust Section */}
      <section style={{ 
        padding: 'var(--spacing-3xl) 0', 
        background: 'var(--text-primary)', 
        color: 'var(--bih-white)',
      }}>
        <div className="container">
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 'var(--spacing-2xl)', textAlign: 'center' }}>
            {[
              { value: '11+', label: 'Languages Supported' },
              { value: '100%', label: 'Local Processing' },
              { value: 'GDPR', label: 'Compliant Design' },
              { value: 'Open', label: 'Source Available' },
            ].map((stat, i) => (
              <div key={i}>
                <div style={{ 
                  fontSize: 'clamp(2rem, 4vw, var(--font-size-3xl))', 
                  fontWeight: 700, 
                  color: 'var(--bih-coral)',
                  marginBottom: 'var(--spacing-sm)',
                }}>
                  {stat.value}
                </div>
                <div style={{ fontSize: 'var(--font-size-lg)', opacity: 0.9 }}>
                  {stat.label}
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Partner Section - With Logos Above Footer */}
      <section style={{ 
        padding: 'var(--spacing-3xl) 0', 
        background: 'var(--bg-secondary)',
        textAlign: 'center',
        borderTop: '1px solid var(--border-color)',
      }}>
        <div className="container">
          <h3 style={{ 
            fontSize: 'var(--font-size-2xl)', 
            fontWeight: 700, 
            color: 'var(--text-primary)', 
            marginBottom: 'var(--spacing-xl)',
          }}>
            Developed By
          </h3>
          <div style={{ 
            display: 'flex', 
            gap: 'var(--spacing-3xl)', 
            justifyContent: 'center', 
            alignItems: 'center', 
            flexWrap: 'wrap',
            marginBottom: 'var(--spacing-xl)',
          }}>
            <div>
              <img src={bihLogoPath} alt="BIH Center" style={{ height: '4rem', width: 'auto', objectFit: 'contain', marginBottom: 'var(--spacing-md)' }} />
              <p style={{ fontSize: 'var(--font-size-lg)', color: 'var(--text-primary)', fontWeight: 600 }}>BIH Center</p>
              <p style={{ fontSize: 'var(--font-size-sm)', color: 'var(--text-muted)' }}>Charité Berlin</p>
            </div>
            <div style={{ fontSize: '2rem', color: 'var(--bih-coral)', fontWeight: 700 }}>&</div>
            <div>
              <img src={dfkiLogoPath} alt="DFKI" style={{ height: '3.5rem', width: 'auto', objectFit: 'contain', marginBottom: 'var(--spacing-md)' }} />
              <p style={{ fontSize: 'var(--font-size-lg)', color: 'var(--text-primary)', fontWeight: 600 }}>DFKI GmbH</p>
              <p style={{ fontSize: 'var(--font-size-sm)', color: 'var(--text-muted)' }}>Kaiserslautern</p>
            </div>
          </div>
          <p style={{ color: 'var(--text-muted)', fontStyle: 'italic' }}>
            Joint initiative under the VERANDA project
          </p>
        </div>
      </section>

      {/* CTA Section */}
      <section style={{ 
        padding: 'var(--spacing-3xl) 0', 
        background: 'var(--bih-coral)',
        color: 'var(--bih-white)', 
        textAlign: 'center',
        position: 'relative',
        overflow: 'hidden',
      }}>
        <div style={{ 
          position: 'absolute', top: 0, left: 0, right: 0, bottom: 0,
          opacity: 0.1,
          backgroundImage: 'radial-gradient(circle at 2px 2px, var(--bih-white) 1px, transparent 0)',
          backgroundSize: '40px 40px',
          zIndex: 0,
        }} />
        
        <div style={{ 
          maxWidth: '800px', 
          margin: '0 auto', 
          padding: '0 var(--spacing-lg)',
          position: 'relative',
          zIndex: 1,
        }}>
          <h2 style={{ 
            fontSize: 'var(--font-size-4xl)', 
            marginBottom: 'var(--spacing-md)',
            fontWeight: 700,
          }}>
            Ready to Anonymize Your Speech Data?
          </h2>
          <p style={{ 
            fontSize: 'var(--font-size-xl)', 
            marginBottom: 'var(--spacing-2xl)', 
            opacity: 0.95,
            lineHeight: 1.7,
          }}>
            Join researchers at leading institutions protecting sensitive information 
            while advancing scientific discovery
          </p>
          <Link to="/signin" className="btn btn-primary btn-lg" style={{
            background: 'var(--bih-white)',
            color: 'var(--bih-coral)',
            boxShadow: '0 10px 40px rgba(0,0,0,0.2)',
          }}>
            Start Your First Job
          </Link>
        </div>
      </section>

      {/* Footer */}
      <footer style={{ 
        background: '#0a1f2e', 
        color: 'var(--bih-white)', 
        padding: 'var(--spacing-3xl) 0 var(--spacing-xl)',
        borderTop: '1px solid rgba(255,255,255,0.1)',
      }}>
        <div className="container">
          <div style={{ 
            display: 'grid', 
            gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', 
            gap: 'var(--spacing-xl)',
            marginBottom: 'var(--spacing-2xl)',
          }}>
            <div>
              <h4 style={{ fontWeight: 700, marginBottom: 'var(--spacing-md)', color: 'var(--bih-coral)' }}>Project</h4>
              <ul style={{ listStyle: 'none', padding: 0, margin: 0 }}>
                <li style={{ marginBottom: 'var(--spacing-sm)' }}>
                  <a href="/app/" style={{ color: 'rgba(255,255,255,0.7)', textDecoration: 'none' }}>Home</a>
                </li>
                <li style={{ marginBottom: 'var(--spacing-sm)' }}>
                  <a href="/app/signin" style={{ color: 'rgba(255,255,255,0.7)', textDecoration: 'none' }}>Sign In</a>
                </li>
                <li style={{ marginBottom: 'var(--spacing-sm)' }}>
                  <a href="/app/dashboard" style={{ color: 'rgba(255,255,255,0.7)', textDecoration: 'none' }}>Dashboard</a>
                </li>
              </ul>
            </div>
            <div>
              <h4 style={{ fontWeight: 700, marginBottom: 'var(--spacing-md)', color: 'var(--bih-coral)' }}>Documentation</h4>
              <ul style={{ listStyle: 'none', padding: 0, margin: 0 }}>
                <li style={{ marginBottom: 'var(--spacing-sm)' }}>
                  <a href="#" style={{ color: 'rgba(255,255,255,0.7)', textDecoration: 'none' }}>User Guide</a>
                </li>
                <li style={{ marginBottom: 'var(--spacing-sm)' }}>
                  <a href="#" style={{ color: 'rgba(255,255,255,0.7)', textDecoration: 'none' }}>API Reference</a>
                </li>
                <li style={{ marginBottom: 'var(--spacing-sm)' }}>
                  <a href="#" style={{ color: 'rgba(255,255,255,0.7)', textDecoration: 'none' }}>Privacy Policy</a>
                </li>
              </ul>
            </div>
            <div>
              <h4 style={{ fontWeight: 700, marginBottom: 'var(--spacing-md)', color: 'var(--bih-coral)' }}>Contact</h4>
              <ul style={{ listStyle: 'none', padding: 0, margin: 0 }}>
                <li style={{ marginBottom: 'var(--spacing-sm)' }}>
                  <span style={{ color: 'rgba(255,255,255,0.7)' }}>Charité Berlin</span>
                </li>
                <li style={{ marginBottom: 'var(--spacing-sm)' }}>
                  <span style={{ color: 'rgba(255,255,255,0.7)' }}>BIH Center</span>
                </li>
                <li style={{ marginBottom: 'var(--spacing-sm)' }}>
                  <span style={{ color: 'rgba(255,255,255,0.7)' }}>DFKI GmbH</span>
                </li>
              </ul>
            </div>
          </div>
          
          <hr style={{ 
            border: 'none', 
            height: '1px', 
            background: 'rgba(255,255,255,0.1)', 
            marginBottom: 'var(--spacing-xl)',
          }} />
          
          <div style={{ textAlign: 'center' }}>
            <p style={{ fontSize: 'var(--font-size-sm)', opacity: 0.5 }}>
              © 2026 VERANDA Project • Charité Berlin • All rights reserved
            </p>
            <p style={{ fontSize: 'var(--font-size-xs)', opacity: 0.4, marginTop: 'var(--spacing-sm)' }}>
              This tool is for research purposes only. Please consult ethics guidelines before processing human subject data.
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default LandingPage;
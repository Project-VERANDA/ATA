import React from 'react';
import { Link } from 'react-router-dom';
import BIHLogo from '../components/BIHLogo';

function LandingPage() {
  const features = [
    { icon: '🎙️', title: 'Speech Transcription', desc: 'WhisperX with speaker diarization for accurate multi-speaker transcription.' },
    { icon: '🔒', title: 'PII Anonymization', desc: 'Local BERT model detects and replaces sensitive information across 11 languages.' },
    { icon: '✨', title: 'LLM Rewriting', desc: 'Advanced LLMs generalize indirect identifiers BERT might miss.' },
    { icon: '🔊', title: 'Synthetic Speech', desc: 'Generate TTS output with multiple voice options for anonymized audio.' },
    { icon: '📁', title: 'Batch Processing', desc: 'Upload and process multiple files with configurable settings.' },
    { icon: '📊', title: 'Job Management', desc: 'Track progress and manage all anonymization jobs from one dashboard.' },
  ];

  return (
    <div style={{ minHeight: '100vh' }}>
      {/* Hero */}
      <section style={{
        background: 'linear-gradient(135deg, var(--bih-blue) 0%, var(--bih-blue-70) 100%)',
        color: 'var(--bih-white)', padding: 'var(--spacing-3xl) 0', position: 'relative', overflow: 'hidden',
      }}>
        <div className="container" style={{ textAlign: 'center', position: 'relative', zIndex: 1 }}>
          <div style={{ display: 'inline-block', marginBottom: 'var(--spacing-lg)' }}>
            <div style={{ color: 'var(--bih-coral)', fontWeight: 600, fontSize: '0.9rem', textTransform: 'uppercase', letterSpacing: '1px' }}>
              Powered by BIH
            </div>
          </div>
          <h1 style={{ fontSize: 'var(--font-size-5xl)', fontWeight: 700, marginBottom: 'var(--spacing-md)' }}>
            Speech <span style={{ color: 'var(--bih-coral)' }}>Anonymizer</span>
          </h1>
          <p style={{ fontSize: 'var(--font-size-xl)', maxWidth: '700px', margin: '0 auto var(--spacing-3xl)', opacity: 0.9 }}>
            Record, upload, transcribe, edit, and anonymize speech with advanced AI models. Protect privacy while preserving data utility for research.
          </p>
          <div style={{ display: 'flex', gap: 'var(--spacing-lg)', justifyContent: 'center', flexWrap: 'wrap' }}>
            <Link to="/signin" className="btn btn-primary btn-lg">Get Started →</Link>
            <a href="#features" className="btn btn-outline btn-lg" style={{ color: 'white', borderColor: 'white' }}>Learn More</a>
          </div>
        </div>
      </section>

      {/* Features */}
      <section id="features" style={{ padding: 'var(--spacing-3xl) 0', background: 'var(--bg-secondary)' }}>
        <div className="container">
          <div style={{ textAlign: 'center', marginBottom: 'var(--spacing-3xl)' }}>
            <h2 style={{ fontSize: 'var(--font-size-4xl)', fontWeight: 700, color: 'var(--text-primary)' }}>Powerful Features</h2>
            <p style={{ fontSize: 'var(--font-size-lg)', color: 'var(--text-muted)', marginTop: 'var(--spacing-sm)' }}>
              Everything you need for comprehensive speech anonymization
            </p>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: 'var(--spacing-xl)' }}>
            {features.map((f, i) => (
              <div key={i} className="fade-in" style={{
                background: 'var(--card-bg)', border: '1px solid var(--border-color)',
                borderRadius: 'var(--radius-lg)', padding: 'var(--spacing-2xl)',
                transition: 'all 0.3s ease',
              }}>
                <div style={{ fontSize: '2.5rem', marginBottom: 'var(--spacing-md)' }}>{f.icon}</div>
                <h3 style={{ fontSize: 'var(--font-size-xl)', color: 'var(--text-primary)', marginBottom: 'var(--spacing-sm)' }}>{f.title}</h3>
                <p style={{ color: 'var(--text-muted)' }}>{f.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA */}
      <section style={{ padding: 'var(--spacing-3xl) 0', background: 'var(--bih-coral)', color: 'white', textAlign: 'center' }}>
        <div className="container">
          <h2 style={{ fontSize: 'var(--font-size-3xl)', marginBottom: 'var(--spacing-md)' }}>Ready to anonymize your speech data?</h2>
          <p style={{ fontSize: 'var(--font-size-lg)', marginBottom: 'var(--spacing-xl)', opacity: 0.9 }}>Join researchers protecting sensitive information</p>
          <Link to="/signin" className="btn btn-primary btn-lg" style={{ background: 'var(--bih-blue)' }}>Start Your First Job</Link>
        </div>
      </section>
    </div>
  );
}

export default LandingPage;

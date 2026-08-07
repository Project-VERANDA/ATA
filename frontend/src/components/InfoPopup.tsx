import React from 'react';

interface InfoPopupProps {
  onClose: () => void;
}

function InfoPopup({ onClose }: InfoPopupProps) {
  return (
    <div style={{
      position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.7)', zIndex: 1050,
      display: 'flex', alignItems: 'center', justifyContent: 'center',
    }}>
      <div style={{
        background: 'var(--card-bg)', borderRadius: 'var(--radius-xl)', maxWidth: '600px',
        width: '90%', maxHeight: '85vh', overflowY: 'auto', animation: 'slideUp 0.3s ease',
      }}>
        <div style={{
          display: 'flex', justifyContent: 'space-between', alignItems: 'center',
          padding: 'var(--spacing-xl)',
          background: 'linear-gradient(135deg, var(--bih-blue) 0%, var(--bih-blue-70) 100%)',
          color: 'var(--bih-white)',
          borderRadius: 'var(--radius-xl) var(--radius-xl) 0 0',
        }}>
          <h2>Welcome to Speech Anonymizer 🎉</h2>
          <button onClick={onClose} style={{ background: 'transparent', fontSize: '1.5rem', color: 'white' }}>×</button>
        </div>

        <div style={{ padding: 'var(--spacing-xl)' }}>
          <p style={{ marginBottom: 'var(--spacing-lg)' }}>
            This is a comprehensive speech anonymization platform developed by the{' '}
            <strong>Berlin Institute of Health (BIH)</strong>.
          </p>

          <h3 style={{ color: 'var(--bih-coral)', margin: 'var(--spacing-lg) 0 var(--spacing-sm)' }}>Key Features:</h3>
          <ul style={{ paddingLeft: 'var(--spacing-xl)', lineHeight: 1.8, color: 'var(--text-secondary)' }}>
            <li>Multi-speaker transcription with diarization</li>
            <li>Local BERT-based PII anonymization (11 languages)</li>
            <li>LLM-powered indirect identifier removal</li>
            <li>Synthetic speech generation (Piper TTS)</li>
            <li>Batch processing support</li>
            <li>Comprehensive audit logging</li>
          </ul>

          <div style={{
            background: 'rgba(234, 84, 81, 0.1)', borderLeft: '4px solid var(--bih-coral)',
            padding: 'var(--spacing-md) var(--spacing-lg)', borderRadius: 'var(--radius-sm)',
            margin: 'var(--spacing-lg) 0', fontSize: '0.9rem', color: 'var(--text-secondary)',
          }}>
            <strong>Important:</strong> All processing is performed locally. No audio data leaves your secure environment.
          </div>

          <h3 style={{ color: 'var(--bih-coral)', margin: 'var(--spacing-lg) 0 var(--spacing-sm)' }}>Getting Started:</h3>
          <ol style={{ paddingLeft: 'var(--spacing-xl)', lineHeight: 1.8, color: 'var(--text-secondary)' }}>
            <li><strong>Submit a Job:</strong> Upload files and configure settings</li>
            <li><strong>Monitor Progress:</strong> Track all your jobs from the Dashboard</li>
            <li><strong>Download Results:</strong> Access anonymized transcripts and synthetic audio</li>
          </ol>
        </div>

        <div style={{ padding: 'var(--spacing-xl)', borderTop: '1px solid var(--border-color)', display: 'flex', justifyContent: 'flex-end' }}>
          <button onClick={onClose} className="btn btn-primary">Got it!</button>
        </div>
      </div>
    </div>
  );
}

export default InfoPopup;

import React, { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';

function JobDetailPage() {
  const { id } = useParams<{ id: string }>();
  const [activeTab, setActiveTab] = useState('overview');

  const job = {
    id, status: 'completed', createdAt: '2026-08-01T10:30:00Z', completedAt: '2026-08-01T10:45:00Z',
    settings: { language: 'EN', whisperModel: 'base', enableDiarization: true, enableAnonymization: true, enableLLMRewrite: false, enableTTS: true },
    files: { input: ['interview_001.mp3'], output: ['anonymized_transcript.txt', 'synthetic_audio.wav'] },
    stats: { duration: '15 min', speakers: 4, piiRemoved: 127 },
  };

  const tabs = ['overview', 'files', 'settings', 'logs'];

  return (
    <div style={{ minHeight: '100vh', padding: 'var(--spacing-2xl) 0' }}>
      <div className="container">
        <div style={{ marginBottom: 'var(--spacing-md)' }}>
          <Link to="/dashboard" style={{ color: 'var(--text-muted)' }}>← Dashboard</Link>
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 'var(--spacing-2xl)' }}>
          <div>
            <h1 style={{ fontSize: 'var(--font-size-3xl)' }}>Job {job.id}</h1>
            <span style={{ color: 'var(--success-color)', fontWeight: 600 }}>● {job.status}</span>
          </div>
          <button className="btn btn-primary">Download All</button>
        </div>

        <div style={{ display: 'flex', gap: 'var(--spacing-sm)', marginBottom: 'var(--spacing-lg)' }}>
          {tabs.map(tab => (
            <button key={tab} onClick={() => setActiveTab(tab)} style={{
              padding: 'var(--spacing-sm) var(--spacing-lg)', borderRadius: 'var(--radius-md)',
              background: activeTab === tab ? 'var(--bih-blue)' : 'transparent',
              color: activeTab === tab ? 'white' : 'var(--text-muted)',
              fontWeight: 600, border: '1px solid var(--border-color)',
            }}>{tab.charAt(0).toUpperCase() + tab.slice(1)}</button>
          ))}
        </div>

        <div style={{ background: 'var(--card-bg)', borderRadius: 'var(--radius-lg)', border: '1px solid var(--border-color)', padding: 'var(--spacing-xl)' }}>
          {activeTab === 'overview' && (
            <div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 'var(--spacing-lg)' }}>
                <div>
                  <h3 style={{ marginBottom: 'var(--spacing-md)' }}>Timing</h3>
                  <p>Created: {job.createdAt}</p>
                  <p>Duration: {job.stats.duration}</p>
                </div>
                <div>
                  <h3 style={{ marginBottom: 'var(--spacing-md)' }}>Results</h3>
                  <p>Speakers: {job.stats.speakers}</p>
                  <p>PII Removed: {job.stats.piiRemoved}</p>
                </div>
              </div>
              <div style={{ marginTop: 'var(--spacing-xl)' }}>
                <h3 style={{ marginBottom: 'var(--spacing-md)' }}>Transcript Preview</h3>
                <pre style={{ background: 'var(--bg-secondary)', padding: 'var(--spacing-md)', borderRadius: 'var(--radius-md)', overflowX: 'auto', fontSize: '0.85rem' }}>
{`SPEAKER_00: Good morning, thank you for joining this meeting.

SPEAKER_01: Thanks for having us. I wanted to discuss the [PROFESSION] certification.

SPEAKER_00: Absolutely. We'll cover all aspects related to [LOCATION_CITY] operations.`}
                </pre>
              </div>
            </div>
          )}
          {activeTab === 'files' && (
            <div>
              <h3 style={{ marginBottom: 'var(--spacing-md)' }}>Input Files</h3>
              {job.files.input.map((f, i) => (
                <div key={i} style={{ display: 'flex', justifyContent: 'space-between', padding: 'var(--spacing-sm) 0', borderBottom: '1px solid var(--border-color)' }}>
                  <span>{f}</span>
                  <button style={{ background: 'transparent' }}>⬇️</button>
                </div>
              ))}
              <h3 style={{ margin: 'var(--spacing-lg) 0 var(--spacing-md)' }}>Output Files</h3>
              {job.files.output.map((f, i) => (
                <div key={i} style={{ display: 'flex', justifyContent: 'space-between', padding: 'var(--spacing-sm) 0', borderBottom: '1px solid var(--border-color)' }}>
                  <span>{f}</span>
                  <button style={{ background: 'transparent' }}>⬇️</button>
                </div>
              ))}
            </div>
          )}
          {activeTab === 'settings' && (
            <div>
              <h3 style={{ marginBottom: 'var(--spacing-md)' }}>Configuration</h3>
              {Object.entries(job.settings).map(([k, v]) => (
                <div key={k} style={{ display: 'flex', justifyContent: 'space-between', padding: 'var(--spacing-sm) 0', borderBottom: '1px solid var(--border-color)' }}>
                  <span style={{ color: 'var(--text-muted)' }}>{k}</span>
                  <span style={{ fontWeight: 600 }}>{String(v)}</span>
                </div>
              ))}
            </div>
          )}
          {activeTab === 'logs' && (
            <div>
              <h3 style={{ marginBottom: 'var(--spacing-md)' }}>Processing Logs</h3>
              <pre style={{ background: 'var(--bg-secondary)', padding: 'var(--spacing-md)', borderRadius: 'var(--radius-md)', fontSize: '0.85rem', fontFamily: 'monospace' }}>
{`[10:30:00] Job started
[10:30:05] Audio files validated (1 file)
[10:30:10] Transcription started (Whisper base, EN)
[10:35:20] Transcription completed
[10:35:25] Diarization started (4 speakers detected)
[10:36:30] Diarization completed
[10:36:35] BERT anonymization started
[10:40:15] PII items identified: 127
[10:42:00] BERT anonymization completed
[10:42:05] TTS generation started
[10:44:50] TTS completed
[10:45:00] Job completed successfully`}
              </pre>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default JobDetailPage;

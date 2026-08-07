import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { useNavigate } from 'react-router-dom';
import type { SubmitFormData } from '../types';

function SubmitJobPage({ onJobSubmitted }: { onJobSubmitted: (jobId: string) => void }) {
  const navigate = useNavigate();
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [formData, setFormData] = useState<SubmitFormData>({
    files: [], folderPath: '', language: 'auto', whisperModel: 'base',
    enableDiarization: true, enableAnonymization: true, anonymizationLevel: 'standard',
    includeTags: ['PERSON', 'ORG', 'LOC_CITY', 'DATETIME', 'CODE_PHONE', 'CODE_URL'],
    enableLLMRewrite: false, llmModel: 'medgemma', enableAdversarial: false,
    enableTTS: true, ttsVoice: 'en_US-amy-medium', batchSize: 10,
  });

  const onDrop = useCallback((accepted: File[]) => {
    setFormData(prev => ({ ...prev, files: [...prev.files, ...accepted] }));
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'audio/*': ['.mp3', '.wav', '.webm', '.ogg'], 'video/*': ['.mp4', '.mkv', '.mov'], 'text/plain': ['.txt'] },
    multiple: true,
  });

  const toggleTag = (tag: string) => {
    setFormData(prev => ({
      ...prev,
      includeTags: prev.includeTags.includes(tag)
        ? prev.includeTags.filter(t => t !== tag)
        : [...prev.includeTags, tag],
    }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);
    await new Promise(r => setTimeout(r, 1500));
    const jobId = 'job-' + Date.now();
    setIsSubmitting(false);
    onJobSubmitted(jobId);
    navigate(`/job/${jobId}`);
  };

  return (
    <div style={{ minHeight: '100vh', padding: 'var(--spacing-2xl) 0' }}>
      <div className="container">
        <h1 style={{ fontSize: 'var(--font-size-4xl)', marginBottom: 'var(--spacing-sm)' }}>Submit Anonymization Job</h1>
        <p style={{ color: 'var(--text-muted)', marginBottom: 'var(--spacing-2xl)' }}>Configure and submit your speech anonymization task</p>

        <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: 'var(--spacing-2xl)' }}>
          {/* File Upload */}
          <section style={{ background: 'var(--card-bg)', border: '1px solid var(--border-color)', borderRadius: 'var(--radius-lg)', padding: 'var(--spacing-xl)' }}>
            <h2 style={{ marginBottom: 'var(--spacing-md)' }}>📁 Files</h2>
            <div {...getRootProps()} style={{
              border: `2px dashed ${isDragActive ? 'var(--bih-coral)' : 'var(--input-border)'}`,
              borderRadius: 'var(--radius-md)', padding: 'var(--spacing-2xl)',
              textAlign: 'center', cursor: 'pointer', transition: 'all 0.3s',
              background: 'var(--input-bg)',
            }}>
              <input {...getInputProps()} />
              <div style={{ fontSize: '3rem', marginBottom: 'var(--spacing-sm)' }}>📂</div>
              <p>{isDragActive ? 'Drop files here...' : 'Drag & drop audio, video, or text files here'}</p>
              <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginTop: 'var(--spacing-xs)' }}>Supports: MP3, WAV, MP4, TXT</p>
            </div>

            {formData.files.length > 0 && (
              <div style={{ marginTop: 'var(--spacing-md)' }}>
                <h3 style={{ marginBottom: 'var(--spacing-sm)' }}>Selected Files ({formData.files.length})</h3>
                {formData.files.map((file, i) => (
                  <div key={i} style={{
                    display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                    padding: 'var(--spacing-sm) var(--spacing-md)', marginBottom: 'var(--spacing-xs)',
                    background: 'var(--hover-bg)', borderRadius: 'var(--radius-sm)',
                  }}>
                    <span style={{ fontSize: '0.9rem' }}>{file.name}</span>
                    <button type="button" onClick={() => setFormData(prev => ({ ...prev, files: prev.files.filter(f => f.name !== file.name) }))} style={{ background: 'transparent', color: 'var(--error-color)', fontSize: '1.2rem' }}>×</button>
                  </div>
                ))}
              </div>
            )}

            <div style={{ marginTop: 'var(--spacing-md)' }}>
              <label style={{ display: 'block', marginBottom: 'var(--spacing-sm)', fontWeight: 600 }}>Or specify folder path:</label>
              <input type="text" value={formData.folderPath} onChange={e => setFormData(prev => ({ ...prev, folderPath: e.target.value }))} placeholder="/path/to/files" style={{
                width: '100%', padding: 'var(--spacing-md)', border: '2px solid var(--input-border)',
                borderRadius: 'var(--radius-md)', background: 'var(--input-bg)', color: 'var(--text-primary)',
              }} />
            </div>
          </section>

          {/* Transcription */}
          <section style={{ background: 'var(--card-bg)', border: '1px solid var(--border-color)', borderRadius: 'var(--radius-lg)', padding: 'var(--spacing-xl)' }}>
            <h2 style={{ marginBottom: 'var(--spacing-md)' }}>🎙️ Transcription</h2>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 'var(--spacing-md)' }}>
              <div>
                <label style={{ display: 'block', marginBottom: 'var(--spacing-sm)', fontWeight: 600 }}>Language:</label>
                <select value={formData.language} onChange={e => setFormData(prev => ({ ...prev, language: e.target.value }))} style={{
                  width: '100%', padding: 'var(--spacing-md)', border: '2px solid var(--input-border)',
                  borderRadius: 'var(--radius-md)', background: 'var(--input-bg)', color: 'var(--text-primary)',
                }}>
                  <option value="auto">Auto-Detect</option>
                  <option value="DE">German</option>
                  <option value="EN">English</option>
                  <option value="FR">French</option>
                  <option value="ES">Spanish</option>
                  <option value="IT">Italian</option>
                  <option value="AR">Arabic</option>
                  <option value="HI">Hindi</option>
                  <option value="TR">Turkish</option>
                </select>
              </div>
              <div>
                <label style={{ display: 'block', marginBottom: 'var(--spacing-sm)', fontWeight: 600 }}>Whisper Model:</label>
                <select value={formData.whisperModel} onChange={e => setFormData(prev => ({ ...prev, whisperModel: e.target.value }))} style={{
                  width: '100%', padding: 'var(--spacing-md)', border: '2px solid var(--input-border)',
                  borderRadius: 'var(--radius-md)', background: 'var(--input-bg)', color: 'var(--text-primary)',
                }}>
                  <option value="tiny">Tiny (Fastest)</option>
                  <option value="base">Base (Default)</option>
                  <option value="small">Small</option>
                  <option value="medium">Medium</option>
                  <option value="large">Large (Best)</option>
                </select>
              </div>
            </div>
            <label style={{ display: 'flex', alignItems: 'center', gap: 'var(--spacing-sm)', marginTop: 'var(--spacing-md)', cursor: 'pointer' }}>
              <input type="checkbox" checked={formData.enableDiarization} onChange={e => setFormData(prev => ({ ...prev, enableDiarization: e.target.checked }))} />
              Enable Speaker Diarization
            </label>
          </section>

          {/* Anonymization */}
          <section style={{ background: 'var(--card-bg)', border: '1px solid var(--border-color)', borderRadius: 'var(--radius-lg)', padding: 'var(--spacing-xl)' }}>
            <h2 style={{ marginBottom: 'var(--spacing-md)' }}>🔒 Anonymization</h2>
            <label style={{ display: 'flex', alignItems: 'center', gap: 'var(--spacing-sm)', marginBottom: 'var(--spacing-md)', cursor: 'pointer' }}>
              <input type="checkbox" checked={formData.enableAnonymization} onChange={e => setFormData(prev => ({ ...prev, enableAnonymization: e.target.checked }))} />
              Enable BERT Anonymization
            </label>
            {formData.enableAnonymization && (
              <>
                <label style={{ fontWeight: 600, display: 'block', marginBottom: 'var(--spacing-sm)' }}>PII Tags to Anonymize:</label>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 'var(--spacing-sm)', marginBottom: 'var(--spacing-md)' }}>
                  {['PERSON', 'PERSON_EMAIL', 'ORG', 'LOC_CITY', 'LOC_COUNTRY', 'LOC_STREET', 'DATETIME', 'DATETIME_AGE', 'CODE_PHONE', 'CODE_URL', 'PROFESSION', 'PRODUCT'].map(tag => (
                    <button key={tag} type="button" onClick={() => toggleTag(tag)} style={{
                      padding: 'var(--spacing-sm) var(--spacing-md)', borderRadius: 'var(--radius-full)',
                      border: `2px solid ${formData.includeTags.includes(tag) ? 'var(--bih-coral)' : 'var(--border-color)'}`,
                      background: formData.includeTags.includes(tag) ? 'rgba(234, 84, 81, 0.15)' : 'transparent',
                      color: formData.includeTags.includes(tag) ? 'var(--bih-coral)' : 'var(--text-muted)',
                      fontWeight: 600, fontSize: '0.85rem', cursor: 'pointer',
                    }}>{tag}</button>
                  ))}
                </div>
              </>
            )}
          </section>

          {/* LLM */}
          <section style={{ background: 'var(--card-bg)', border: '1px solid var(--border-color)', borderRadius: 'var(--radius-lg)', padding: 'var(--spacing-xl)' }}>
            <h2 style={{ marginBottom: 'var(--spacing-md)' }}>✨ LLM Processing</h2>
            <label style={{ display: 'flex', alignItems: 'center', gap: 'var(--spacing-sm)', marginBottom: 'var(--spacing-md)', cursor: 'pointer' }}>
              <input type="checkbox" checked={formData.enableLLMRewrite} onChange={e => setFormData(prev => ({ ...prev, enableLLMRewrite: e.target.checked }))} />
              Enable LLM Rewrite
            </label>
            {formData.enableLLMRewrite && (
              <>
                <div style={{ marginBottom: 'var(--spacing-md)' }}>
                  <label style={{ display: 'block', marginBottom: 'var(--spacing-sm)', fontWeight: 600 }}>LLM Model:</label>
                  <select value={formData.llmModel} onChange={e => setFormData(prev => ({ ...prev, llmModel: e.target.value }))} style={{
                    width: '100%', padding: 'var(--spacing-md)', border: '2px solid var(--input-border)',
                    borderRadius: 'var(--radius-md)', background: 'var(--input-bg)', color: 'var(--text-primary)',
                  }}>
                    <option value="medgemma">MedGemma</option>
                    <option value="medgemma27b">MedGemma 27B</option>
                    <option value="gpt-oss-120b">GPT OSS 120B</option>
                    <option value="Qwen3.6-27B">Qwen3.6-27B</option>
                  </select>
                </div>
                <label style={{ display: 'flex', alignItems: 'center', gap: 'var(--spacing-sm)', cursor: 'pointer' }}>
                  <input type="checkbox" checked={formData.enableAdversarial} onChange={e => setFormData(prev => ({ ...prev, enableAdversarial: e.target.checked }))} />
                  Enable Adversarial Mode (3 iterations)
                </label>
              </>
            )}
          </section>

          {/* TTS */}
          <section style={{ background: 'var(--card-bg)', border: '1px solid var(--border-color)', borderRadius: 'var(--radius-lg)', padding: 'var(--spacing-xl)' }}>
            <h2 style={{ marginBottom: 'var(--spacing-md)' }}>🔊 Synthetic Speech</h2>
            <label style={{ display: 'flex', alignItems: 'center', gap: 'var(--spacing-sm)', marginBottom: 'var(--spacing-md)', cursor: 'pointer' }}>
              <input type="checkbox" checked={formData.enableTTS} onChange={e => setFormData(prev => ({ ...prev, enableTTS: e.target.checked }))} />
              Generate Synthetic Audio Output
            </label>
            {formData.enableTTS && (
              <div>
                <label style={{ display: 'block', marginBottom: 'var(--spacing-sm)', fontWeight: 600 }}>Voice:</label>
                <select value={formData.ttsVoice} onChange={e => setFormData(prev => ({ ...prev, ttsVoice: e.target.value }))} style={{
                  width: '100%', padding: 'var(--spacing-md)', border: '2px solid var(--input-border)',
                  borderRadius: 'var(--radius-md)', background: 'var(--input-bg)', color: 'var(--text-primary)',
                }}>
                  <option value="en_US-amy-medium">Amy (Female, American)</option>
                  <option value="en_US-lessac-medium">Lessac (Female, High Quality)</option>
                  <option value="en_US-kusal-medium">Kusal (Male, American)</option>
                  <option value="en_US-ryan-medium">Ryan (Male, Deep)</option>
                  <option value="en_US-joe-medium">Joe (Male, Casual)</option>
                </select>
              </div>
            )}
          </section>

          {/* Submit */}
          <div style={{ display: 'flex', gap: 'var(--spacing-md)', justifyContent: 'flex-end' }}>
            <button type="reset" className="btn btn-outline">Reset</button>
            <button type="submit" className="btn btn-primary" disabled={isSubmitting || formData.files.length === 0}>
              {isSubmitting ? (<><span className="loading-spinner"></span> Processing...</>) : 'Submit Job →'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

export default SubmitJobPage;

import React, { useState } from 'react';

interface SurveyPopupProps {
  jobId: string;
  onClose: () => void;
}

function SurveyPopup({ jobId, onClose }: SurveyPopupProps) {
  const [rating, setRating] = useState(0);
  const [feedback, setFeedback] = useState('');
  const [submitted, setSubmitted] = useState(false);

  const handleSubmit = () => {
    setSubmitted(true);
    setTimeout(onClose, 1500);
  };

  if (submitted) {
    return (
      <div style={{
        position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.7)', zIndex: 1050,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
      }}>
        <div style={{
          background: 'var(--card-bg)', borderRadius: 'var(--radius-xl)', padding: 'var(--spacing-3xl)',
          textAlign: 'center', maxWidth: '400px', animation: 'slideUp 0.3s ease',
        }}>
          <div style={{ fontSize: '3rem', marginBottom: 'var(--spacing-md)' }}>🎉</div>
          <h2>Thank You!</h2>
          <p style={{ color: 'var(--text-muted)' }}>Your feedback helps us improve.</p>
        </div>
      </div>
    );
  }

  return (
    <div style={{
      position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.7)', zIndex: 1050,
      display: 'flex', alignItems: 'center', justifyContent: 'center',
    }}>
      <div style={{
        background: 'var(--card-bg)', borderRadius: 'var(--radius-xl)', maxWidth: '500px',
        width: '90%', maxHeight: '80vh', overflowY: 'auto', animation: 'slideUp 0.3s ease',
      }}>
        <div style={{
          display: 'flex', justifyContent: 'space-between', alignItems: 'center',
          padding: 'var(--spacing-xl)', borderBottom: '1px solid var(--border-color)',
        }}>
          <h2>Job Submitted Successfully!</h2>
          <button onClick={onClose} style={{
            background: 'transparent', fontSize: '1.5rem', color: 'var(--text-muted)',
          }}>×</button>
        </div>

        <div style={{ padding: 'var(--spacing-xl)' }}>
          <p style={{ marginBottom: 'var(--spacing-lg)' }}>Job ID: <code>{jobId}</code></p>

          <div style={{ margin: 'var(--spacing-xl) 0' }}>
            <label style={{ fontWeight: 600, display: 'block', marginBottom: 'var(--spacing-md)' }}>
              Rate your experience:
            </label>
            <div style={{ display: 'flex', gap: 'var(--spacing-sm)', justifyContent: 'center' }}>
              {[1, 2, 3, 4, 5].map(star => (
                <button
                  key={star}
                  onClick={() => setRating(star)}
                  style={{
                    background: 'transparent', border: 'none', fontSize: '2rem',
                    color: star <= rating ? 'var(--bih-coral)' : 'var(--text-muted)',
                    cursor: 'pointer',
                  }}
                >★</button>
              ))}
            </div>
          </div>

          <div style={{ marginTop: 'var(--spacing-xl)' }}>
            <label style={{ fontWeight: 600, display: 'block', marginBottom: 'var(--spacing-sm)' }}>
              Additional feedback (optional):
            </label>
            <textarea
              value={feedback}
              onChange={e => setFeedback(e.target.value)}
              placeholder="Tell us what worked well or what we could improve..."
              rows={4}
              style={{
                width: '100%', padding: 'var(--spacing-md)',
                border: '2px solid var(--input-border)', borderRadius: 'var(--radius-md)',
                background: 'var(--input-bg)', color: 'var(--text-primary)',
                fontFamily: 'inherit', resize: 'vertical',
              }}
            />
          </div>
        </div>

        <div style={{
          display: 'flex', justifyContent: 'flex-end', gap: 'var(--spacing-md)',
          padding: 'var(--spacing-xl)', borderTop: '1px solid var(--border-color)',
        }}>
          <button onClick={onClose} className="btn btn-outline">Skip</button>
          <button
            onClick={handleSubmit}
            className="btn btn-primary"
            disabled={rating === 0}
          >Submit Feedback</button>
        </div>
      </div>
    </div>
  );
}

export default SurveyPopup;

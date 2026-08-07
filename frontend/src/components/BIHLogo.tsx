import React from 'react';

interface BIHLogoProps {
  variant?: 'standard' | 'short';
  size?: 'sm' | 'md' | 'lg';
}

function BIHLogo({ variant = 'standard', size = 'md' }: BIHLogoProps) {
  const sizes = {
    sm: { bar: '28px', barH: '3px', text: '1.2em', sub: '0.6em' },
    md: { bar: '40px', barH: '4px', text: '1.5em', sub: '0.7em' },
    lg: { bar: '56px', barH: '5px', text: '2em', sub: '0.8em' },
  };
  const s = sizes[size];

  return (
    <div style={{ display: 'inline-flex', alignItems: 'center', gap: '12px' }}>
      <div style={{ width: s.bar, height: s.barH, background: 'var(--bih-coral)', flexShrink: 0 }} />
      {variant === 'standard' ? (
        <div style={{ display: 'flex', flexDirection: 'column', lineHeight: 1.2 }}>
          <span style={{ fontSize: s.text, fontWeight: 700, color: 'var(--text-primary)' }}>BIH</span>
          <span style={{ fontSize: s.sub, color: 'var(--text-muted)' }}>at Charité</span>
        </div>
      ) : (
        <span style={{ fontSize: s.text, fontWeight: 700, color: 'var(--text-primary)' }}>BIH</span>
      )}
    </div>
  );
}

export default BIHLogo;

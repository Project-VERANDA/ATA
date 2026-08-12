import React from 'react';

interface BIHLogoProps {
  variant?: 'standard' | 'short';
  size?: 'sm' | 'md' | 'lg';
}

function BIHLogo({ variant = 'standard', size = 'md' }: BIHLogoProps) {
  const sizes = {
    sm: { width: '100px', height: 'auto' },
    md: { width: '150px', height: 'auto' },
    lg: { width: '200px', height: 'auto' },
  };
  const s = sizes[size];

  return (
    <a href="/app/" style={{ textDecoration: 'none', display: 'inline-flex', alignItems: 'center' }}>
      <img 
        src="/logos/Online_251121A_BIH_Logo_RGB_BIH_Logo_StandardClaim_ENG_BlauKorall.svg"
        alt="BIH at Charité" 
        style={{ 
          width: s.width, 
          height: s.height,
          flexShrink: 0,
          display: 'block'
        }}
      />
    </a>
  );
}

export default BIHLogo;
import React from 'react';

interface BIHLogoProps {
  variant?: 'standard' | 'short';
  size?: 'sm' | 'md' | 'lg';
}

function BIHLogo({ variant = 'standard', size = 'md' }: BIHLogoProps) {
  // Adjust these dimensions based on how the SVG looks in your app
  const sizes = {
    sm: { width: '80px', height: 'auto' },
    md: { width: '120px', height: 'auto' },
    lg: { width: '180px', height: 'auto' },
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

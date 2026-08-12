import React from 'react';

interface LogoProps {
  src: string;
  alt: string;
  height?: string;
}

const Logo: React.FC<LogoProps> = ({ src, alt, height = '40px' }) => (
  <img 
    src={src} 
    alt={alt} 
    style={{ 
      height, 
      width: 'auto', 
      maxHeight: '100%',
      objectFit: 'contain'
    }} 
  />
);

export const BrandingLogos: React.FC<{ variant?: 'landing' | 'footer' }> = ({ variant = 'landing' }) => {
  const containerStyle: React.CSSProperties = {
    display: 'flex',
    alignItems: 'center',
    justifyContent: variant === 'landing' ? 'space-between' : 'center',
    flexWrap: 'wrap',
    gap: '20px',
    padding: '20px 0',
  };

  const logoContainerStyle: React.CSSProperties = {
    display: 'flex',
    alignItems: 'center',
    gap: '20px',
  };

  return (
    <div style={containerStyle}>
      <div style={logoContainerStyle}>
        {/* BIH Logo */}
        <Logo 
          src="/logos/Online_251121A_BIH_Logo_RGB_BIH_Logo_StandardClaim_ENG_BlauKorall.svg" 
          alt="Berlin Institute of Health (BIH) Logo" 
          height="60px"
        />
        {/* DFKI Logo */}
        <Logo 
          src="/logos/dfki-logo_sz.svg" 
          alt="German Research Center for Artificial Intelligence (DFKI) Logo" 
          height="50px"
        />
      </div>
      
      {variant === 'landing' && (
        <div style={{ textAlign: 'right' }}>
          <small style={{ color: '#666' }}>
            Joint Initiative
          </small>
        </div>
      )}
    </div>
  );
};

export default BrandingLogos;

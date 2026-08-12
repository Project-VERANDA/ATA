import React from 'react';
import bihLogo from '/logos/Online_251121A_BIH_Logo_RGB_BIH_Logo_StandardClaim_ENG_BlauKorall.svg';
import dfkiLogo from '/logos/dfki_Logo_sz.svg';
import verandaLogo from '/logos/VERANDA_LOGO.svg';

interface BrandingLogosProps {
  variant?: 'hero' | 'footer' | 'compact';
  showEquation?: boolean;
  className?: string;
}

const BrandingLogos: React.FC<BrandingLogosProps> = ({
  variant = 'compact',
  showEquation = true,
  className = '',
}) => {
  const sizes = {
    hero: { height: '4rem', gap: '2.5rem', textSize: '2.5rem' },
    footer: { height: '3rem', gap: '3rem', textSize: '2rem' },
    compact: { height: '2rem', gap: '1.5rem', textSize: '1.25rem' },
  };

  const size = sizes[variant];

  return (
    <div className={className}>
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: size.gap,
        justifyContent: 'center',
        flexWrap: 'wrap',
      }}>
        <img 
          src={bihLogo} 
          alt="BIH Center" 
          style={{ 
            height: size.height, 
            width: 'auto', 
            objectFit: 'contain',
            transition: 'opacity 0.3s ease',
          }}
          onError={(e) => {
            console.warn('BIH logo failed to load:', e);
          }}
        />
        
        {showEquation && (
          <>
            <span style={{ fontSize: size.textSize, fontWeight: 700, opacity: 0.5 }}>+</span>
            <img 
              src={dfkiLogo} 
              alt="DFKI" 
              style={{ 
                height: size.height * 0.85, 
                width: 'auto', 
                objectFit: 'contain',
              }}
              onError={(e) => {
                console.warn('DFKI logo failed to load:', e);
              }}
            />
            <span style={{ fontSize: size.textSize, fontWeight: 700, opacity: 0.5 }}>=</span>
            <img 
              src={verandaLogo} 
              alt="VERANDA" 
              style={{ 
                height: size.height * 1.2, 
                width: 'auto', 
                objectFit: 'contain',
              }}
              onError={(e) => {
                console.warn('VERANDA logo failed to load:', e);
              }}
            />
          </>
        )}
        
        {!showEquation && (
          <img 
            src={verandaLogo} 
            alt="VERANDA" 
            style={{ 
              height: size.height * 1.2, 
              width: 'auto', 
              objectFit: 'contain',
            }}
          />
        )}
      </div>
    </div>
  );
};

export default BrandingLogos;
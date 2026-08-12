import React from 'react';

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
  const bihLogoPath = '/logos/Online_251121A_BIH_Logo_RGB_BIH_Logo_StandardClaim_ENG_BlauKorall.svg';
  const dfkiLogoPath = '/logos/dfki_Logo_sz.svg';
  const verandaLogoPath = '/logos/VERANDA_LOGO.svg';

  const sizes: Record<'hero' | 'footer' | 'compact', { height: number; gap: string; textSize: string }> = {
    hero: { height: 64, gap: '2.5rem', textSize: '2.5rem' },
    footer: { height: 48, gap: '3rem', textSize: '2rem' },
    compact: { height: 32, gap: '1.5rem', textSize: '1.25rem' },
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
          src={bihLogoPath} 
          alt="BIH Center" 
          style={{ 
            height: `${size.height}px`, 
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
              src={dfkiLogoPath} 
              alt="DFKI" 
              style={{ 
                height: `${size.height * 0.85}px`, 
                width: 'auto', 
                objectFit: 'contain',
              }}
              onError={(e) => {
                console.warn('DFKI logo failed to load:', e);
              }}
            />
            <span style={{ fontSize: size.textSize, fontWeight: 700, opacity: 0.5 }}>=</span>
            <img 
              src={verandaLogoPath} 
              alt="VERANDA" 
              style={{ 
                height: `${size.height * 1.2}px`, 
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
            src={verandaLogoPath} 
            alt="VERANDA" 
            style={{ 
              height: `${size.height * 1.2}px`, 
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
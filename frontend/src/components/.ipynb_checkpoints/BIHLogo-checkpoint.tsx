import React from 'react';

interface BIHLogoProps {
  variant?: 'standard' | 'short';
  size?: 'sm' | 'md' | 'lg';
  className?: string;
}

const BIHLogo: React.FC<BIHLogoProps> = ({
  variant = 'standard',
  size = 'md',
  className = '',
}) => {
  // Use public asset path (actual BIH logo SVG file)
  const bihLogoPath = '/logos/Online_251121A_BIH_Logo_RGB_BIH_Logo_StandardClaim_ENG_BlauKorall.svg';

  const sizes = {
    sm: { height: '2rem' },
    md: { height: '3rem' },
    lg: { height: '4rem' },
  };

  const sizeStyles = sizes[size];

  return (
    <div 
      className={className} 
      style={{ 
        display: 'inline-block',
        lineHeight: 0, // Remove extra space below image
      }}
    >
      <img 
        src={bihLogoPath} 
        alt="BIH Center at Charité" 
        style={{ 
          height: sizeStyles.height, 
          width: 'auto', 
          objectFit: 'contain',
          display: 'block',
        }}
        onError={(e) => {
          console.error('BIH logo failed to load:', e);
          // Show fallback text if image fails
          const target = e.target as HTMLImageElement;
          target.style.display = 'none';
          const fallback = target.parentElement?.querySelector('.bih-logo-fallback');
          if (fallback) {
            fallback.style.display = 'inline-block';
          }
        }}
      />
      {/* Fallback if logo image fails to load */}
      <span 
        className="bih-logo-fallback"
        style={{
          display: 'none',
          fontSize: size === 'lg' ? '1.5rem' : size === 'md' ? '1.25rem' : '1rem',
          fontWeight: 700,
          color: 'var(--text-primary)',
          marginLeft: '8px',
        }}
      >
        BIH<span style={{ color: 'var(--bih-coral)' }}>Center</span>
      </span>
    </div>
  );
};

export default BIHLogo;
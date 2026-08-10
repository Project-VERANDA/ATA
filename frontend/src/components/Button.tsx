import React from 'react';

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'outline' | 'danger';
  size?: 'sm' | 'md' | 'lg';
  isLoading?: boolean;
  leftIcon?: React.ReactNode;
  rightIcon?: React.ReactNode;
  fullWidth?: boolean;
  children: React.ReactNode;
}

const Button: React.FC<ButtonProps> = ({
  variant = 'primary',
  size = 'md',
  isLoading = false,
  leftIcon,
  rightIcon,
  fullWidth = false,
  disabled,
  style,
  children,
  ...props
}) => {
  // Corner radius for rectangular buttons with smooth edges
  const CORNER_RADIUS = '4px';

  const baseStyles: React.CSSProperties = {
    border: 'none',
    borderRadius: CORNER_RADIUS,
    cursor: disabled || isLoading ? 'not-allowed' : 'pointer',
    transition: 'all 0.2s ease',
    fontWeight: 500,
    display: 'inline-flex',
    alignItems: 'center',
    justifyContent: 'center',
    gap: '8px',
    opacity: disabled || isLoading ? 0.6 : 1,
  };

  const variants = {
    primary: {
      backgroundColor: '#6d4aff',
      color: 'white',
      '&:hover:not(:disabled)': {
        backgroundColor: '#5a40e8',
      },
    },
    secondary: {
      backgroundColor: '#f3f4f6',
      color: '#374151',
      '&:hover:not(:disabled)': {
        backgroundColor: '#e5e7eb',
      },
    },
    outline: {
      backgroundColor: 'transparent',
      color: '#6d4aff',
      border: '2px solid #6d4aff',
      '&:hover:not(:disabled)': {
        backgroundColor: '#f0f4ff',
      },
    },
    danger: {
      backgroundColor: '#ef4444',
      color: 'white',
      '&:hover:not(:disabled)': {
        backgroundColor: '#dc2626',
      },
    },
  };

  const sizes = {
    sm: { padding: '6px 12px', fontSize: '12px' },
    md: { padding: '10px 20px', fontSize: '14px' },
    lg: { padding: '14px 28px', fontSize: '16px' },
  };

  const computedStyle: React.CSSProperties = {
    ...baseStyles,
    ...(variants[variant] as React.CSSProperties),
    ...(sizes[size]),
    ...(fullWidth ? { width: '100%' } : {}),
    ...(style || {}),
  };

  // Inline styles need explicit hover states via pseudo-element workaround
  // For simplicity, we'll use a style object (in production, consider CSS modules or styled-components)

  return (
    <button
      style={computedStyle}
      disabled={disabled || isLoading}
      {...props}
    >
      {isLoading && (
        <span style={{
          width: '14px',
          height: '14px',
          border: '2px solid currentColor',
          borderTopColor: 'transparent',
          borderRadius: '50%',
          animation: 'spin 1s linear infinite',
        }} />
      )}
      {!isLoading && leftIcon && <span>{leftIcon}</span>}
      <span>{children}</span>
      {!isLoading && rightIcon && <span>{rightIcon}</span>}
    </button>
  );
};

export default Button;

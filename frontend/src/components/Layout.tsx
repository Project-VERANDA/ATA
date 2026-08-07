import React, { useContext, useState } from 'react';
import { Link, useLocation, Outlet, Navigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { useTheme } from '../contexts/ThemeContext';
import BIHLogo from './BIHLogo';
import { useAuth as useAuthHook } from '../contexts/AuthContext';

function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const { isAuthenticated, isLoading } = useAuthHook();
  if (isLoading) return <div>Loading...</div>;
  return isAuthenticated ? <>{children}</> : <Navigate to="/signin" />;
}

export { ProtectedRoute };

function Layout() {
  const { user, logout, isAuthenticated } = useAuth();
  const { theme, toggleTheme } = useTheme();
  const location = useLocation();
  const [menuOpen, setMenuOpen] = useState(false);

  const isAuthPage = location.pathname === '/signin';

  const navLinks = [
    { name: 'Home', path: '/' },
    ...(isAuthenticated ? [
      { name: 'Submit Job', path: '/submit-job' },
      { name: 'Dashboard', path: '/dashboard' },
    ] : []),
  ];

  if (isAuthPage) {
    return (
      <div className={theme === 'dark' ? 'dark-theme' : ''}>
        <Outlet />
      </div>
    );
  }

  return (
    <div className={theme === 'dark' ? 'dark-theme' : ''} style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      {/* Navbar */}
      <nav style={{
        background: 'var(--bg-primary)',
        borderBottom: `2px solid var(--border-color)`,
        padding: 'var(--spacing-md) var(--spacing-xl)',
        boxShadow: 'var(--shadow-sm)',
        position: 'sticky',
        top: 0,
        zIndex: 100,
      }}>
        <div style={{ maxWidth: '1200px', margin: '0 auto', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <Link to="/" style={{ textDecoration: 'none' }}>
            <BIHLogo variant="standard" size="md" />
          </Link>

          <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--spacing-lg)' }}>
            {navLinks.map(link => (
              <Link
                key={link.path}
                to={link.path}
                style={{
                  color: location.pathname === link.path ? 'var(--bih-coral)' : 'var(--text-primary)',
                  fontWeight: 600,
                  textDecoration: 'none',
                  padding: 'var(--spacing-sm) var(--spacing-md)',
                  borderRadius: 'var(--radius-md)',
                  transition: 'all 0.2s',
                }}
              >
                {link.name}
              </Link>
            ))}

            <button
              onClick={toggleTheme}
              style={{
                background: 'transparent',
                border: '1px solid var(--border-color)',
                borderRadius: 'var(--radius-full)',
                width: '36px',
                height: '36px',
                fontSize: '1.2rem',
                cursor: 'pointer',
              }}
              aria-label="Toggle theme"
            >
              {theme === 'light' ? '🌙' : '☀️'}
            </button>

            {isAuthenticated ? (
              <div style={{ position: 'relative' }}>
                <button
                  onClick={() => setMenuOpen(!menuOpen)}
                  style={{
                    background: 'var(--bih-blue)',
                    color: 'var(--bih-white)',
                    borderRadius: 'var(--radius-full)',
                    width: '36px',
                    height: '36px',
                    fontWeight: 700,
                  }}
                >
                  {user?.name?.charAt(0) || 'U'}
                </button>
                {menuOpen && (
                  <div style={{
                    position: 'absolute',
                    right: 0,
                    top: '44px',
                    background: 'var(--card-bg)',
                    border: '1px solid var(--border-color)',
                    borderRadius: 'var(--radius-md)',
                    boxShadow: 'var(--shadow-lg)',
                    minWidth: '200px',
                    padding: 'var(--spacing-sm)',
                    zIndex: 200,
                  }}>
                    <div style={{ padding: 'var(--spacing-md)', borderBottom: '1px solid var(--border-color)', marginBottom: 'var(--spacing-sm)' }}>
                      <div style={{ fontWeight: 600 }}>{user?.name}</div>
                      <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>{user?.email}</div>
                    </div>
                    <button
                      onClick={() => { logout(); setMenuOpen(false); }}
                      style={{
                        width: '100%',
                        textAlign: 'left',
                        padding: 'var(--spacing-md)',
                        background: 'transparent',
                        borderRadius: 'var(--radius-sm)',
                        color: 'var(--error-color)',
                      }}
                    >
                      Sign Out
                    </button>
                  </div>
                )}
              </div>
            ) : (
              <Link to="/signin" className="btn btn-primary" style={{ padding: 'var(--spacing-sm) var(--spacing-lg)' }}>
                Sign In
              </Link>
            )}
          </div>
        </div>
      </nav>

      {/* Main content */}
      <main style={{ flex: 1 }}>
        <Outlet />
      </main>

      {/* Footer */}
      <footer style={{
        background: 'var(--bih-blue)',
        color: 'var(--bih-white)',
        padding: 'var(--spacing-2xl) var(--spacing-xl)',
        marginTop: 'auto',
      }}>
        <div style={{ maxWidth: '1200px', margin: '0 auto', textAlign: 'center' }}>
          <BIHLogo variant="short" size="sm" />
          <p style={{ marginTop: 'var(--spacing-md)', opacity: 0.7, fontSize: '0.85rem' }}>
            © {new Date().getFullYear()} Berlin Institute of Health at Charité · VERANDA Project
          </p>
        </div>
      </footer>
    </div>
  );
}

export default Layout;

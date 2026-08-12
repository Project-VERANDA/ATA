import React, { createContext, useState, useContext, useEffect } from 'react';
import type { AuthContextType, User } from '../types';

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const token = localStorage.getItem('kc_token');
    const userInfo = localStorage.getItem('kc_user');
    if (token && userInfo) {
      setUser(JSON.parse(userInfo));
      setIsAuthenticated(true);
    }
    setIsLoading(false);
  }, []);

  const login = async () => {
    const mockUser: User = {
      id: 'demo-user-001',
      name: 'Demo Researcher',
      email: 'demo@bih.charite.de',
      avatar: null,
    };
    localStorage.setItem('kc_user', JSON.stringify(mockUser));
    localStorage.setItem('kc_token', 'mock-token-' + Date.now());
    setUser(mockUser);
    setIsAuthenticated(true);
  };

  const logout = async () => {
    localStorage.removeItem('kc_token');
    localStorage.removeItem('kc_user');
    setUser(null);
    setIsAuthenticated(false);
  };

  return (
    <AuthContext.Provider value={{ user, isAuthenticated, isLoading, login, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be used within AuthProvider');
  return ctx;
}

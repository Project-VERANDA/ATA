import React, { useState, useEffect } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { AuthProvider } from './contexts/AuthContext';
import { ThemeProvider } from './contexts/ThemeContext';
import Layout, { ProtectedRoute } from './components/Layout';
import LandingPage from './pages/LandingPage';
import SignInPage from './pages/SignInPage';
import SubmitJobPage from './pages/SubmitJobPage';
import DashboardPage from './pages/DashboardPage';
import JobDetailPage from './pages/JobDetailPage';
import SurveyPopup from './components/SurveyPopup';
import InfoPopup from './components/InfoPopup';

function App() {
  const [showSurvey, setShowSurvey] = useState(false);
  const [surveyJobId, setSurveyJobId] = useState<string | null>(null);
  const [showInfo, setShowInfo] = useState(() => !localStorage.getItem('infoSeen'));

  useEffect(() => {
    if (!showInfo) return;
    const handler = setTimeout(() => {}, 100);
    return () => clearTimeout(handler);
  }, [showInfo]);

  return (
    <ThemeProvider>
      <AuthProvider>
          <BrowserRouter basename="/app"> 
            <Routes>
              <Route element={<Layout />}>
                <Route path="/" element={<LandingPage />} />
                <Route path="/signin" element={<SignInPage />} />
                <Route
                  path="/submit-job"
                  element={
                    <ProtectedRoute>
                      <SubmitJobPage onJobSubmitted={(jobId) => {
                        setSurveyJobId(jobId);
                        setShowSurvey(true);
                      }} />
                    </ProtectedRoute>
                  }
                />
                <Route path="/dashboard" element={<ProtectedRoute><DashboardPage /></ProtectedRoute>} />
                <Route path="/job/:id" element={<ProtectedRoute><JobDetailPage /></ProtectedRoute>} />
              </Route>
            </Routes>
    
            {showSurvey && surveyJobId && (
              <SurveyPopup jobId={surveyJobId} onClose={() => { setShowSurvey(false); setSurveyJobId(null); }} />
            )}
    
            {showInfo && (
              <InfoPopup onClose={() => { setShowInfo(false); localStorage.setItem('infoSeen', 'true'); }} />
            )}
          </BrowserRouter>
      </AuthProvider>
    </ThemeProvider>
  );
}

export default App;

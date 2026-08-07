import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import type { Job } from '../types';

function DashboardPage() {
  const [filter, setFilter] = useState<string>('all');

  const mockJobs: Job[] = [
    { id: 'job-001', status: 'completed', createdAt: '2026-08-01', files: 3, type: 'audio' },
    { id: 'job-002', status: 'processing', createdAt: '2026-08-02', files: 5, type: 'video' },
    { id: 'job-003', status: 'completed', createdAt: '2026-07-28', files: 2, type: 'text' },
    { id: 'job-004', status: 'pending', createdAt: '2026-08-03', files: 10, type: 'batch' },
  ];

  const jobs = mockJobs.filter(j => filter === 'all' || j.status === filter);

  const stats = {
    total: mockJobs.length,
    completed: mockJobs.filter(j => j.status === 'completed').length,
    processing: mockJobs.filter(j => j.status === 'processing').length,
    pending: mockJobs.filter(j => j.status === 'pending').length,
  };

  const statusColors: Record<string, string> = {
    completed: 'var(--success-color)', processing: 'var(--warning-color)',
    pending: 'var(--text-muted)', failed: 'var(--error-color)',
  };

  return (
    <div style={{ minHeight: '100vh', padding: 'var(--spacing-2xl) 0' }}>
      <div className="container">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 'var(--spacing-2xl)' }}>
          <h1 style={{ fontSize: 'var(--font-size-4xl)' }}>Job Dashboard</h1>
          <Link to="/submit-job" className="btn btn-primary">+ New Job</Link>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 'var(--spacing-md)', marginBottom: 'var(--spacing-2xl)' }}>
          {[
            { label: 'Total Jobs', value: stats.total, icon: '📊' },
            { label: 'Completed', value: stats.completed, icon: '✅' },
            { label: 'Processing', value: stats.processing, icon: '⏳' },
            { label: 'Pending', value: stats.pending, icon: '⏱️' },
          ].map((s, i) => (
            <div key={i} style={{ background: 'var(--card-bg)', border: '1px solid var(--border-color)', borderRadius: 'var(--radius-lg)', padding: 'var(--spacing-lg)' }}>
              <div style={{ fontSize: '1.5rem', marginBottom: 'var(--spacing-sm)' }}>{s.icon}</div>
              <div style={{ fontSize: 'var(--font-size-2xl)', fontWeight: 700 }}>{s.value}</div>
              <div style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>{s.label}</div>
            </div>
          ))}
        </div>

        <div style={{ display: 'flex', gap: 'var(--spacing-sm)', marginBottom: 'var(--spacing-lg)' }}>
          {['all', 'completed', 'processing', 'pending'].map(tab => (
            <button key={tab} onClick={() => setFilter(tab)} style={{
              padding: 'var(--spacing-sm) var(--spacing-lg)',
              borderRadius: 'var(--radius-full)',
              background: filter === tab ? 'var(--bih-coral)' : 'transparent',
              color: filter === tab ? 'white' : 'var(--text-muted)',
              border: `1px solid ${filter === tab ? 'var(--bih-coral)' : 'var(--border-color)'}`,
              fontWeight: 600,
            }}>{tab.charAt(0).toUpperCase() + tab.slice(1)}</button>
          ))}
        </div>

        <div style={{ background: 'var(--card-bg)', borderRadius: 'var(--radius-lg)', border: '1px solid var(--border-color)', overflow: 'hidden' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ background: 'var(--bg-secondary)' }}>
                <th style={{ padding: 'var(--spacing-md)', textAlign: 'left', borderBottom: '1px solid var(--border-color)' }}>Job ID</th>
                <th style={{ padding: 'var(--spacing-md)', textAlign: 'left', borderBottom: '1px solid var(--border-color)' }}>Status</th>
                <th style={{ padding: 'var(--spacing-md)', textAlign: 'left', borderBottom: '1px solid var(--border-color)' }}>Files</th>
                <th style={{ padding: 'var(--spacing-md)', textAlign: 'left', borderBottom: '1px solid var(--border-color)' }}>Type</th>
                <th style={{ padding: 'var(--spacing-md)', textAlign: 'left', borderBottom: '1px solid var(--border-color)' }}>Created</th>
              </tr>
            </thead>
            <tbody>
              {jobs.map(job => (
                <tr key={job.id} style={{ borderBottom: '1px solid var(--border-color)' }}>
                  <td style={{ padding: 'var(--spacing-md)' }}>
                    <Link to={`/job/${job.id}`} style={{ fontFamily: 'monospace', color: 'var(--bih-coral)' }}>{job.id}</Link>
                  </td>
                  <td style={{ padding: 'var(--spacing-md)' }}>
                    <span style={{ color: statusColors[job.status], fontWeight: 600 }}>● {job.status}</span>
                  </td>
                  <td style={{ padding: 'var(--spacing-md)' }}>{job.files}</td>
                  <td style={{ padding: 'var(--spacing-md)' }}><span style={{ textTransform: 'uppercase', fontSize: '0.8rem', fontWeight: 600 }}>{job.type}</span></td>
                  <td style={{ padding: 'var(--spacing-md)' }}>{job.createdAt}</td>
                </tr>
              ))}
              {jobs.length === 0 && (
                <tr><td colSpan={5} style={{ padding: 'var(--spacing-xl)', textAlign: 'center', color: 'var(--text-muted)' }}>
                  No jobs found. <Link to="/submit-job">Submit your first job</Link>
                </td></tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

export default DashboardPage;

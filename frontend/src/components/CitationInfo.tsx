import React, { useState } from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faCopy, faCheck } from '@fortawesome/free-solid-svg-icons';

const CitationInfo: React.FC = () => {
  const [copied, setCopied] = useState<'none' | 'bibtex' | 'text'>('none');

  const bibtexEntry = `@software{BHSpeechAnonymizer2026,
  author = {BIH Speech Anonymization Team},
  title = {BIH Speech Anonymization Tool},
  year = {2026},
  institution = {Berlin Institute of Health (BIH)},
  address = {Berlin, Germany},
  url = {https://transcriber.cloud.cci.charite.de},
  license = {Proprietary - Contact authors for access}
}`;

  const plainCitation = `Dialogue Anonymization (2026). Berlin Institute of Health (BIH) & Deutsches Forschungszentrum für Künstliche Intelligenz GmbH (DFKI), Berlin, Germany. Available at: https://github.com/Project-VERANDA/ATA/tree/main`;

  const copyToClipboard = async (text: string, type: 'bibtex' | 'text') => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(type);
      setTimeout(() => setCopied('none'), 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };

  const CopyButton: React.FC<{ text: string; type: 'bibtex' | 'text' }> = ({ text, type }) => (
    <button
      onClick={() => copyToClipboard(text, type)}
      style={{
        backgroundColor: copied === type ? '#4CAF50' : '#6d4aff',
        color: 'white',
        border: 'none',
        borderRadius: '4px',
        padding: '8px 16px',
        cursor: 'pointer',
        display: 'flex',
        alignItems: 'center',
        gap: '8px',
        marginTop: '8px',
        fontSize: '14px',
      }}
    >
      <FontAwesomeIcon icon={copied === type ? faCheck : faCopy} />
      {copied === type ? 'Copied!' : 'Copy'}
    </button>
  );

  return (
    <div style={{ padding: '24px', maxWidth: '700px' }}>
      {/* Header */}
      <div style={{ marginBottom: '20px' }}>
        <h3 style={{ margin: '0 0 12px', color: '#6d4aff' }}>
          📊 Speech Anonymization Tool
        </h3>
        <p style={{ margin: 0, color: '#666', fontSize: '14px' }}>
          Developed jointly by <strong>Berlin Institute of Health (BIH)</strong> and 
          <strong> German Research Center for Artificial Intelligence (DFKI)</strong>
        </p>
      </div>

      {/* Citation Box */}
      <div style={{ 
        backgroundColor: '#f8f9fa', 
        borderLeft: '4px solid #6d4aff', 
        padding: '16px', 
        borderRadius: '4px',
        marginBottom: '20px'
      }}>
        <h4 style={{ margin: '0 0 12px', fontSize: '16px' }}>📝 Citation Required</h4>
        <p style={{ margin: '0 0 16px', fontSize: '14px', lineHeight: '1.5' }}>
          If you use this tool in scientific publications, please cite it using one of the formats below:
        </p>
        
        <div style={{ marginBottom: '12px' }}>
          <details style={{ fontSize: '14px' }}>
            <summary style={{ cursor: 'pointer', fontWeight: 'bold' }}>APA Style (Click to expand)</summary>
            <p style={{ margin: '8px 0 0', padding: '8px', backgroundColor: '#fff', borderRadius: '4px' }}>
              BIH Speech Anonymization Tool (2026). Berlin Institute of Health (BIH), Berlin, Germany.
            </p>
          </details>
        </div>

        <div style={{ marginBottom: '12px' }}>
          <details style={{ fontSize: '14px' }}>
            <summary style={{ cursor: 'pointer', fontWeight: 'bold' }}>BibTeX (Click to expand)</summary>
            <textarea
              readOnly
              value={bibtexEntry}
              style={{
                width: '100%',
                minHeight: '100px',
                margin: '8px 0',
                padding: '8px',
                fontFamily: 'monospace',
                fontSize: '12px',
                border: '1px solid #ddd',
                borderRadius: '4px',
                resize: 'vertical',
              }}
            />
            <CopyButton text={bibtexEntry} type="bibtex" />
          </details>
        </div>

        <div>
          <details style={{ fontSize: '14px' }}>
            <summary style={{ cursor: 'pointer', fontWeight: 'bold' }}>Plain Text (Click to expand)</summary>
            <textarea
              readOnly
              value={plainCitation}
              style={{
                width: '100%',
                minHeight: '60px',
                margin: '8px 0',
                padding: '8px',
                fontFamily: 'monospace',
                fontSize: '12px',
                border: '1px solid #ddd',
                borderRadius: '4px',
                resize: 'horizontal',
              }}
            />
            <CopyButton text={plainCitation} type="text" />
          </details>
        </div>
      </div>

      {/* Acknowledgments */}
      <div style={{ marginBottom: '20px' }}>
        <h4 style={{ margin: '0 0 12px', fontSize: '16px' }}>Acknowledgments</h4>
        <ul style={{ margin: 0, paddingLeft: '20px', fontSize: '14px', lineHeight: '1.6' }}>
          <li>This tool was developed as part of the VERANDA project between AG Health Data Privacy at the Berlin Institute of Health at the Charité and Dr. Roland Roller at the Speech and Language Group at the German Center for Artificial Intelligence (Deutsches Forschungszentrum für Künstliche Intelligenz GmbH, DFKI) .</li>
          <li></li>
          <li></li>
        </ul>
      </div>

      {/* Funding */}
      <div style={{ marginBottom: '20px' }}>
        <h4 style={{ margin: '0 0 12px', fontSize: '16px' }}>Funding</h4>
        <p style={{ margin: 0, fontSize: '14px', lineHeight: '1.6' }}>
          This research and tools has been supported by the German Federal Ministry of Education and 
Research (BMBF) through project VERANDA (16K1S2046K).
        </p>
      </div>

      {/* Contact Information */}
      <div style={{ 
        backgroundColor: '#f0f4ff', 
        padding: '16px', 
        borderRadius: '4px'
      }}>
        <h4 style={{ margin: '0 0 12px', fontSize: '16px' }}>Contact Information</h4>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '14px' }}>
          <tbody>
            <tr>
              <td style={{ padding: '6px 0', fontWeight: 'bold', width: '120px' }}>Technical:</td>
              <td style={{ padding: '6px 0' }}>
                <a href="mailto:luke.flanagan@bih-charite.de" style={{ color: '#6d4aff' }}>luke.flanagan@bih-charite.de</a>
              </td>
            </tr>
            <tr>
              <td style={{ padding: '6px 0', fontWeight: 'bold' }}>Scientific:</td>
              <td style={{ padding: '6px 0' }}>
                <a href="mailto:gds-studien@bih-charite.de" style={{ color: '#6d4aff' }}>gds-studien@addrebih-charite.de</a>
              </td>
            </tr>
            <tr>
              <td style={{ padding: '6px 0', fontWeight: 'bold' }}>Institution:</td>
              <td style={{ padding: '6px 0' }}>
                AG Poikela<br/>
		Berlin Institute of Health at Charité<br/>
		Center of Health Data Sciences<br/>
		Charitéplatz 1<br/> 
		10117 Berlin <br/>
                </td>
            </tr>
          </tbody>
        </table>
      </div>

      {/* Footer Links */}
      <div style={{ marginTop: '20px', borderTop: '1px solid #eee', paddingTop: '16px' }}>
        <p style={{ margin: '0 0 8px', fontSize: '14px' }}>
          🔒 <strong>Data Privacy:</strong> All audio processing occurs on secure servers. 
          No data is stored permanently without explicit consent.
        </p>
        <p style={{ margin: 0, fontSize: '12px', color: '#888' }}>
          Last updated: August 2026 | Version 1.0.0
        </p>
      </div>
    </div>
  );
};

export default CitationInfo;

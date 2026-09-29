import React, { useState, useEffect, useRef } from 'react';
import styles from './App.module.css';
import { ConstellationCanvas } from './components/ConstellationCanvas';
import { TickRing } from './components/TickRing';
import { CustomCursor } from './components/CustomCursor';
import { ANCHORS_CONFIG } from './components/RoleAnchors';
import type {
  RoleName,
  WordWeight,
  PredictResponse,
  MetadataResponse,
  SceneState,
} from './types';
import {
  checkApiHealth,
  fetchMetadata,
  fetchSamples,
  sendPrediction,
} from './api';

export const App: React.FC = () => {
  // Application & Scene State
  const [sceneState, setSceneState] = useState<SceneState>('landing');
  const [isApiLive, setIsApiLive] = useState(false);
  const [resumeText, setResumeText] = useState('');
  const [targetRole, setTargetRole] = useState<string>('');
  const [validationError, setValidationError] = useState<string | null>(null);

  // Tabs: 'none' | 'how-it-works' | 'model-results' (hidden by default)
  const [activeTab, setActiveTab] = useState<'none' | 'how-it-works' | 'model-results'>('none');

  // Result details toggle ("See why")
  const [showDetails, setShowDetails] = useState(false);

  // Custom Dropdown state
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Results State
  const [prediction, setPrediction] = useState<PredictResponse | null>(null);
  const [hoveredRole, setHoveredRole] = useState<RoleName | null>(null);
  const [hoveredWord, setHoveredWord] = useState<WordWeight | null>(null);

  // Metadata & Samples
  const [metadata, setMetadata] = useState<MetadataResponse | null>(null);
  const [samples, setSamples] = useState<Record<string, string>>({});
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  // Initialize and check API
  useEffect(() => {
    const init = async () => {
      const healthy = await checkApiHealth();
      setIsApiLive(healthy);

      const [metaData, sampleData] = await Promise.all([
        fetchMetadata(),
        fetchSamples(),
      ]);

      if (metaData) setMetadata(metaData);
      if (sampleData) setSamples(sampleData);
    };

    init();
    const interval = setInterval(async () => {
      const healthy = await checkApiHealth();
      setIsApiLive(healthy);
    }, 10000);

    return () => clearInterval(interval);
  }, []);

  // Close dropdown on outside click
  useEffect(() => {
    const handleOutsideClick = (e: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setIsDropdownOpen(false);
      }
    };
    document.addEventListener('mousedown', handleOutsideClick);
    return () => document.removeEventListener('mousedown', handleOutsideClick);
  }, []);

  // Word count helper
  const wordCount = resumeText.trim()
    ? resumeText.trim().split(/\s+/).length
    : 0;

  // Handle Text change
  const handleTextChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const text = e.target.value;
    setResumeText(text);
    if (validationError) setValidationError(null);

    if (sceneState !== 'analyzing') {
      setSceneState(text.trim() ? 'typing' : 'landing');
    }
  };

  // Load sample resume profile
  const handleLoadSample = (sampleKey: string) => {
    if (samples[sampleKey]) {
      setResumeText(samples[sampleKey]);
      setValidationError(null);
      setSceneState('typing');
    }
  };

  // Clear text
  const handleClear = () => {
    setResumeText('');
    setPrediction(null);
    setValidationError(null);
    setSceneState('landing');
  };

  // Execute Analysis
  const handleAnalyze = async () => {
    if (!resumeText.trim()) {
      setValidationError('Please paste or type resume text before analyzing.');
      return;
    }
    if (wordCount < 20) {
      setValidationError(
        `Resume has only ${wordCount} words. Please provide at least 20 words for an accurate prediction.`
      );
      return;
    }

    setValidationError(null);
    setIsAnalyzing(true);
    setSceneState('analyzing');

    try {
      const result = await sendPrediction(resumeText, targetRole || undefined);

      setTimeout(() => {
        setPrediction(result);
        setSceneState('result');
        setIsAnalyzing(false);

        const resElem = document.getElementById('results-view');
        if (resElem) {
          resElem.scrollIntoView({ behavior: 'smooth' });
        }
      }, 750);
    } catch (err: unknown) {
      setIsAnalyzing(false);
      setSceneState('typing');
      const msg = err instanceof Error ? err.message : 'Analysis failed';
      setValidationError(`Server error: ${msg}`);
    }
  };

  // Reset to input state
  const handleReset = () => {
    setPrediction(null);
    setShowDetails(false);
    setSceneState(resumeText.trim() ? 'typing' : 'landing');
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  return (
    <div className={styles.appContainer}>
      {/* Custom Trailing Reticle Cursor */}
      <CustomCursor />

      {/* Background 3D Constellation Canvas (Low opacity until Analyze) */}
      <ConstellationCanvas
        resumeText={resumeText}
        state={sceneState}
        predictedRole={prediction ? prediction.predicted_role : null}
        topWords={prediction ? prediction.top_words : []}
        hoveredRole={hoveredRole}
        onHoverRole={setHoveredRole}
        onHoverWord={setHoveredWord}
        topCoefficients={metadata ? metadata.top_coefficients : undefined}
      />

      {/* Top Minimal Masthead */}
      <header className={styles.masthead}>
        <div className={styles.brandTitle}>Resume to Role</div>

        <div className={styles.mastheadNav}>
          <button
            type="button"
            className={`${styles.tabBtn} ${activeTab === 'how-it-works' ? styles.tabBtnActive : ''}`}
            onClick={() => setActiveTab((prev) => (prev === 'how-it-works' ? 'none' : 'how-it-works'))}
          >
            How it works
          </button>
          <button
            type="button"
            className={`${styles.tabBtn} ${activeTab === 'model-results' ? styles.tabBtnActive : ''}`}
            onClick={() => setActiveTab((prev) => (prev === 'model-results' ? 'none' : 'model-results'))}
          >
            Model results
          </button>
          <div className={styles.apiStatus}>
            <span
              className={styles.statusDot}
              style={{ background: isApiLive ? '#ff4d1a' : '#8a847b' }}
            />
            <span>
              {hoveredWord
                ? `Focus: ${hoveredWord.word}`
                : isApiLive
                ? 'Online'
                : 'Connecting'}
            </span>
          </div>
        </div>
      </header>

      {/* Main Content Layout */}
      <main className={styles.contentArea}>
        {/* ===================================================================
            01. LANDING: HEADLINE + SUBTEXT + RESUME BOX + DROPDOWN + ANALYZE
            Nothing else above the fold.
            =================================================================== */}
        <section className={styles.heroSection}>
          <h1 className={styles.headline}>
            Match any resume to the right tech role.
          </h1>

          <p className={styles.subtext}>
            Paste a resume below to see its best-fit role, match score, and key skills.
          </p>

          <div className={styles.inputCard}>
            {/* Quick Sample Selector */}
            <div className={styles.sampleBar}>
              <span className={styles.sampleLabel}>Try an example:</span>
              {Object.keys(samples).map((sampleKey) => (
                <button
                  key={sampleKey}
                  type="button"
                  className={`${styles.sampleChip} ${
                    samples[sampleKey] === resumeText ? styles.sampleChipActive : ''
                  }`}
                  onClick={() => handleLoadSample(sampleKey)}
                >
                  {sampleKey}
                </button>
              ))}
              {resumeText && (
                <button
                  type="button"
                  className={styles.clearChip}
                  onClick={handleClear}
                >
                  Clear
                </button>
              )}
            </div>

            {/* Resume Input Textarea */}
            <textarea
              className={styles.resumeTextarea}
              placeholder="Paste resume text here..."
              value={resumeText}
              onChange={handleTextChange}
              disabled={isAnalyzing}
              spellCheck={false}
            />

            {validationError && (
              <div className={styles.warningMessage}>
                {validationError}
              </div>
            )}

            {/* Action Row: Dropdown + Analyze Button */}
            <div className={styles.actionRow}>
              {/* Plain Role Dropdown */}
              <div className={styles.roleDropdownWrapper} ref={dropdownRef}>
                <button
                  type="button"
                  className={`${styles.roleDropdownBtn} ${isDropdownOpen ? styles.open : ''}`}
                  onClick={() => setIsDropdownOpen((prev) => !prev)}
                  disabled={isAnalyzing}
                >
                  <span>
                    {targetRole ? `Target: ${targetRole}` : 'Auto-detect best fit'}
                  </span>
                  <span className={styles.dropdownChevron}>▼</span>
                </button>

                {isDropdownOpen && (
                  <div className={styles.dropdownMenu}>
                    <div
                      className={`${styles.dropdownOption} ${!targetRole ? styles.selected : ''}`}
                      onClick={() => {
                        setTargetRole('');
                        setIsDropdownOpen(false);
                      }}
                    >
                      Auto-detect best fit
                    </div>
                    {metadata?.roles.map((r) => (
                      <div
                        key={r}
                        className={`${styles.dropdownOption} ${targetRole === r ? styles.selected : ''}`}
                        onClick={() => {
                          setTargetRole(r);
                          setIsDropdownOpen(false);
                        }}
                      >
                        {r}
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {/* Analyze Button */}
              <button
                type="button"
                className={styles.analyzeButton}
                onClick={handleAnalyze}
                disabled={isAnalyzing}
              >
                {isAnalyzing ? 'Analyzing resume...' : 'Analyze resume'}
              </button>
            </div>
          </div>
        </section>

        {/* ===================================================================
            02. RESULT VIEW: PREDICTED ROLE + FIT SCORE + TOP 5 WORDS
            Details hidden behind "See why"
            =================================================================== */}
        {prediction && (
          <section id="results-view" className={styles.resultSection}>
            <div className={styles.resultCard}>
              {/* Header: Predicted Role & Fit Score */}
              <div className={styles.resultHeaderRow}>
                <div className={styles.resultRoleBlock}>
                  <div className={styles.resultMetaLabel}>Best-fit tech role</div>
                  <h2 className={styles.resultRoleTitle}>{prediction.predicted_role}</h2>
                  <div className={styles.resultCertainty}>
                    Model certainty: {prediction.probabilities[prediction.predicted_role]}%
                  </div>
                </div>

                <div className={styles.scoreGaugeBlock}>
                  <TickRing
                    score={prediction.fit_score}
                    label={`Score: ${prediction.target_role}`}
                    subLabel="Fit score (0–100)"
                  />
                </div>
              </div>

              {/* Top 5 Key Skills */}
              <div className={styles.topWordsSection}>
                <div className={styles.resultMetaLabel}>Top skills driving this match</div>
                <div className={styles.topWordsChips}>
                  {prediction.top_words.slice(0, 5).map((w) => (
                    <div key={w.word} className={styles.skillChip}>
                      <span className={styles.skillWord}>{w.word}</span>
                      <span className={styles.skillWeight}>+{w.weight.toFixed(3)}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* "See why" Toggle Button */}
              <div className={styles.seeWhyContainer}>
                <button
                  type="button"
                  className={styles.seeWhyBtn}
                  onClick={() => setShowDetails((prev) => !prev)}
                >
                  {showDetails ? 'Hide details ▲' : 'See why ▼'}
                </button>
              </div>

              {/* Expandable Details Section behind "See why" */}
              {showDetails && (
                <div className={styles.expandedDetails}>
                  {/* Role Probability Breakdown */}
                  <div className={styles.detailsGroup}>
                    <div className={styles.detailsSubtitle}>Role probability breakdown</div>
                    <div className={styles.probList}>
                      {Object.entries(prediction.probabilities).map(([roleKey, probVal]) => {
                        const isPrimary = roleKey === prediction.predicted_role;
                        return (
                          <div key={roleKey} className={styles.probItem}>
                            <div className={styles.probHeader}>
                              <span style={{ color: isPrimary ? '#ff4d1a' : '#ece7df' }}>
                                {roleKey}
                              </span>
                              <span style={{ color: isPrimary ? '#ff4d1a' : '#8a847b' }}>
                                {probVal.toFixed(1)}%
                              </span>
                            </div>
                            <div className={styles.probTrack}>
                              <div
                                className={`${styles.probFill} ${isPrimary ? styles.active : ''}`}
                                style={{ width: `${probVal}%` }}
                              />
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>

                  {/* Deeper Driving Words Table */}
                  <div className={styles.detailsGroup}>
                    <div className={styles.detailsSubtitle}>Key skill contributions</div>
                    <table className={styles.wordsTable}>
                      <thead>
                        <tr>
                          <th>Skill / Keyword</th>
                          <th style={{ textAlign: 'right' }}>Impact</th>
                          <th style={{ textAlign: 'right' }}>Frequency (TF-IDF)</th>
                        </tr>
                      </thead>
                      <tbody>
                        {prediction.top_words.slice(0, 8).map((w) => (
                          <tr key={w.word}>
                            <td style={{ color: '#ece7df', fontWeight: 500 }}>{w.word}</td>
                            <td style={{ textAlign: 'right', color: '#ff4d1a', fontWeight: 600 }}>
                              +{w.weight.toFixed(4)}
                            </td>
                            <td style={{ textAlign: 'right', color: '#8a847b' }}>
                              {w.tfidf?.toFixed(3) || '—'}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              {/* Reset Button */}
              <div className={styles.resetRow}>
                <button type="button" className={styles.resetBtn} onClick={handleReset}>
                  Analyze another resume
                </button>
              </div>
            </div>
          </section>
        )}
      </main>

      {/* ===================================================================
          03. TWO TABS: "HOW IT WORKS" & "MODEL RESULTS" (Hidden by default)
          =================================================================== */}
      {activeTab !== 'none' && (
        <div className={styles.modalBackdrop} onClick={() => setActiveTab('none')}>
          <div className={styles.tabModal} onClick={(e) => e.stopPropagation()}>
            <div className={styles.modalHeader}>
              <h3 className={styles.modalTitle}>
                {activeTab === 'how-it-works' ? 'How it works' : 'Model results'}
              </h3>
              <button
                type="button"
                className={styles.modalCloseBtn}
                onClick={() => setActiveTab('none')}
              >
                Close ✕
              </button>
            </div>

            <div className={styles.modalBody}>
              {activeTab === 'how-it-works' ? (
                <>
                  <div className={styles.infoSection}>
                    <div className={styles.infoHeading}>The 5 Tech Roles</div>
                    <p className={styles.infoText}>
                      We map engineering skills across five standard disciplines:
                    </p>
                    <div className={styles.roleCardGrid}>
                      {ANCHORS_CONFIG.map((a) => (
                        <div key={a.role} className={styles.roleCard}>
                          <div className={styles.roleCardTitle}>{a.name}</div>
                          <div className={styles.roleCardSkills}>{a.description}</div>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div className={styles.infoSection}>
                    <div className={styles.infoHeading}>How Matching Works</div>
                    <p className={styles.infoText}>
                      1. <strong>Cleaning</strong>: We strip noise while protecting key programming terms like C++, .NET, and Node.js.
                    </p>
                    <p className={styles.infoText}>
                      2. <strong>Keywords & Phrases</strong>: The model identifies both single tools (Python, Docker) and two-word skills (machine learning, data science).
                    </p>
                    <p className={styles.infoText}>
                      3. <strong>Fit Score</strong>: A Ridge regression model evaluates overall alignment on a continuous 0–100 scale.
                    </p>
                  </div>

                  <div className={styles.infoSection}>
                    <div className={styles.infoHeading}>3D Constellation</div>
                    <p className={styles.infoText}>
                      Words from your resume float as points in space. When you analyze, key technical terms fly toward the matching role anchor to visualize what drove the prediction.
                    </p>
                  </div>
                </>
              ) : (
                <>
                  <div className={styles.infoSection}>
                    <div className={styles.infoHeading}>Classification Accuracy (Test Set)</div>
                    <p className={styles.infoText}>
                      Evaluated on a stratified 80/20 test split of deduplicated technical resumes:
                    </p>
                    <table className={styles.specTable}>
                      <thead>
                        <tr>
                          <th>Model</th>
                          <th style={{ textAlign: 'right' }}>Accuracy</th>
                          <th style={{ textAlign: 'right' }}>Macro F1</th>
                        </tr>
                      </thead>
                      <tbody>
                        <tr>
                          <td style={{ color: '#ff4d1a', fontWeight: 600 }}>
                            Logistic Regression (Balanced)
                          </td>
                          <td style={{ textAlign: 'right' }}>90.9%</td>
                          <td style={{ textAlign: 'right' }}>0.7679</td>
                        </tr>
                        <tr>
                          <td>Multinomial Naive Bayes</td>
                          <td style={{ textAlign: 'right' }}>81.8%</td>
                          <td style={{ textAlign: 'right' }}>0.6692</td>
                        </tr>
                      </tbody>
                    </table>
                  </div>

                  <div className={styles.infoSection}>
                    <div className={styles.infoHeading}>Fit Score Accuracy (Ridge vs Probabilities)</div>
                    <p className={styles.infoText}>
                      Average point error (Mean Absolute Error on a 0–100 scale). Lower error is better:
                    </p>
                    <table className={styles.specTable}>
                      <thead>
                        <tr>
                          <th>Role</th>
                          <th style={{ textAlign: 'right' }}>Ridge Error (MAE)</th>
                          <th style={{ textAlign: 'right' }}>Probability Error</th>
                        </tr>
                      </thead>
                      <tbody>
                        <tr>
                          <td>Web Dev</td>
                          <td style={{ textAlign: 'right', color: '#ff4d1a' }}>7.87 pts</td>
                          <td style={{ textAlign: 'right' }}>15.22 pts</td>
                        </tr>
                        <tr>
                          <td>AI / ML</td>
                          <td style={{ textAlign: 'right', color: '#ff4d1a' }}>10.57 pts</td>
                          <td style={{ textAlign: 'right' }}>19.24 pts</td>
                        </tr>
                        <tr>
                          <td>Analyst</td>
                          <td style={{ textAlign: 'right', color: '#ff4d1a' }}>11.24 pts</td>
                          <td style={{ textAlign: 'right' }}>18.67 pts</td>
                        </tr>
                        <tr>
                          <td>Other Tech</td>
                          <td style={{ textAlign: 'right', color: '#ff4d1a' }}>32.31 pts</td>
                          <td style={{ textAlign: 'right' }}>36.43 pts</td>
                        </tr>
                        <tr>
                          <td>Backend</td>
                          <td style={{ textAlign: 'right', color: '#ff4d1a' }}>38.23 pts</td>
                          <td style={{ textAlign: 'right' }}>48.02 pts</td>
                        </tr>
                      </tbody>
                    </table>
                    <p className={styles.infoText} style={{ fontSize: '10.5px', marginTop: '6px' }}>
                      Ridge regression achieves lower point error across all roles compared to raw probabilities.
                    </p>
                  </div>
                </>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default App;

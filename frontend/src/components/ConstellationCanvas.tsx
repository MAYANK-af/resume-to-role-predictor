import React, { Suspense } from 'react';
import { Canvas } from '@react-three/fiber';
import { WordParticles } from './WordParticles';
import { RoleAnchors } from './RoleAnchors';
import { CameraRig } from './CameraRig';
import type { RoleName, WordWeight } from '../types';

interface ConstellationCanvasProps {
  resumeText: string;
  state: string;
  predictedRole: RoleName | null;
  topWords: WordWeight[];
  hoveredRole: RoleName | null;
  onHoverRole: (role: RoleName | null) => void;
  onHoverWord: (word: WordWeight | null) => void;
  topCoefficients?: Record<RoleName, Array<{ display: string; weight: number }>>;
}

export const ConstellationCanvas: React.FC<ConstellationCanvasProps> = ({
  resumeText,
  state,
  predictedRole,
  topWords,
  hoveredRole,
  onHoverRole,
  onHoverWord,
  topCoefficients,
}) => {
  // Lowered ambient opacity before analysis; comes alive once Analyze is clicked
  const isAlive = state === 'analyzing' || state === 'result';
  const canvasOpacity = isAlive ? 0.88 : 0.22;

  return (
    <div
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        width: '100vw',
        height: '100vh',
        zIndex: 0,
        pointerEvents: 'none',
        opacity: canvasOpacity,
        transition: 'opacity 1.2s cubic-bezier(0.16, 1, 0.3, 1)',
      }}
    >
      <Canvas
        camera={{ position: [0, 0, 11], fov: 50, near: 0.1, far: 100 }}
        dpr={Math.min(window.devicePixelRatio, 2)}
        gl={{ antialias: true, alpha: true }}
        style={{ width: '100%', height: '100%', pointerEvents: 'auto' }}
      >
        <ambientLight intensity={0.6} />
        <directionalLight position={[10, 10, 5]} intensity={0.8} />

        <Suspense fallback={null}>
          <RoleAnchors
            predictedRole={predictedRole}
            hoveredRole={hoveredRole}
            onHoverRole={onHoverRole}
            state={state}
            topCoefficients={topCoefficients}
          />

          <WordParticles
            resumeText={resumeText}
            state={state}
            predictedRole={predictedRole}
            topWords={topWords}
            onHoverWord={onHoverWord}
          />

          <CameraRig state={state} predictedRole={predictedRole} />
        </Suspense>
      </Canvas>
    </div>
  );
};

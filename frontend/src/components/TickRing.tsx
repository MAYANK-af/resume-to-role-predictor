import React, { useEffect, useState, useRef } from 'react';
import gsap from 'gsap';

interface TickRingProps {
  score: number;
  label?: string;
  subLabel?: string;
}

export const TickRing: React.FC<TickRingProps> = ({
  score,
  label = 'FIT SCORE',
  subLabel = 'RIDGE REGRESSION',
}) => {
  const [displayScore, setDisplayScore] = useState(0);
  const scoreRef = useRef({ val: 0 });

  const totalTicks = 60;
  const radius = 104;
  const cx = 140;
  const cy = 140;

  useEffect(() => {
    // Non-linear exponential deceleration curve (Zero default GSAP power3)
    gsap.to(scoreRef.current, {
      val: score,
      duration: 1.8,
      ease: 'expo.out',
      onUpdate: () => {
        setDisplayScore(Math.round(scoreRef.current.val));
      },
    });
  }, [score]);

  // Generate 60 precision radial tick marks
  const activeTickCount = Math.round((displayScore / 100) * totalTicks);

  const ticks = Array.from({ length: totalTicks }, (_, i) => {
    // Start at bottom-left (-90 deg offset)
    const angleDeg = (i / totalTicks) * 360 - 90;
    const angleRad = (angleDeg * Math.PI) / 180;

    const isMajor = i % 5 === 0;
    const tickLen = isMajor ? 13 : 7;
    const isActive = i <= activeTickCount;

    const x1 = cx + (radius - tickLen) * Math.cos(angleRad);
    const y1 = cy + (radius - tickLen) * Math.sin(angleRad);
    const x2 = cx + radius * Math.cos(angleRad);
    const y2 = cy + radius * Math.sin(angleRad);

    return {
      id: i,
      x1,
      y1,
      x2,
      y2,
      isActive,
      isMajor,
    };
  });

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        position: 'relative',
      }}
    >
      <svg width="280" height="280" viewBox="0 0 280 280">
        <defs>
          <radialGradient id="ringGlow" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor="rgba(255, 77, 26, 0.08)" />
            <stop offset="100%" stopColor="rgba(255, 77, 26, 0)" />
          </radialGradient>
        </defs>

        {/* Subtle center warm aura */}
        <circle cx={cx} cy={cy} r={radius - 20} fill="url(#ringGlow)" />

        {/* Inner thin technical perimeter track */}
        <circle
          cx={cx}
          cy={cy}
          r={radius - 17}
          fill="none"
          stroke="rgba(236, 231, 223, 0.07)"
          strokeWidth="1"
          strokeDasharray="2 4"
        />

        {/* Outer technical perimeter track */}
        <circle
          cx={cx}
          cy={cy}
          r={radius + 8}
          fill="none"
          stroke="rgba(236, 231, 223, 0.05)"
          strokeWidth="1"
        />

        {/* 60 Precision Technical Tick Marks */}
        {ticks.map((t) => (
          <line
            key={t.id}
            x1={t.x1}
            y1={t.y1}
            x2={t.x2}
            y2={t.y2}
            stroke={t.isActive ? '#ff4d1a' : 'rgba(236, 231, 223, 0.11)'}
            strokeWidth={t.isMajor ? 2.2 : 1.0}
            strokeLinecap="square"
            style={{
              transition: 'stroke 0.35s cubic-bezier(0.16, 1, 0.3, 1)',
            }}
          />
        ))}

        {/* Huge Editorial Serif Score */}
        <text
          x={cx}
          y={cy + 10}
          textAnchor="middle"
          fill="#ece7df"
          fontFamily="'Fraunces', 'Instrument Serif', Georgia, serif"
          fontSize="64"
          fontWeight="300"
          letterSpacing="-0.045em"
        >
          {displayScore}
          <tspan
            fontSize="24"
            fill="#ff4d1a"
            fontFamily="'JetBrains Mono', monospace"
            fontWeight="600"
            dx="3"
          >
            %
          </tspan>
        </text>

        {/* Micro Technical Sublabel */}
        <text
          x={cx}
          y={cy + 38}
          textAnchor="middle"
          fill="#8a847b"
          fontFamily="'JetBrains Mono', monospace"
          fontSize="9.5"
          letterSpacing="0.18em"
          style={{ textTransform: 'uppercase' }}
        >
          {label}
        </text>
      </svg>

      <div
        style={{
          fontFamily: "'JetBrains Mono', monospace",
          fontSize: '10px',
          color: '#8a847b',
          letterSpacing: '0.16em',
          textTransform: 'uppercase',
          marginTop: '-12px',
        }}
      >
        [{subLabel}]
      </div>
    </div>
  );
};

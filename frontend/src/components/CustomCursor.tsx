import React, { useEffect, useState } from 'react';

export const CustomCursor: React.FC = () => {
  const [pos, setPos] = useState({ x: -100, y: -100 });
  const [trail, setTrail] = useState({ x: -100, y: -100 });
  const [isPointer, setIsPointer] = useState(false);
  const [isDown, setIsDown] = useState(false);
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    // Only mount on desktop devices with precision pointers
    if (!window.matchMedia('(hover: hover) and (pointer: fine)').matches) {
      return;
    }

    const handleMouseMove = (e: MouseEvent) => {
      setPos({ x: e.clientX, y: e.clientY });
      setIsVisible(true);

      const target = e.target as HTMLElement | null;
      if (target) {
        const isClickable =
          target.tagName === 'BUTTON' ||
          target.tagName === 'A' ||
          target.tagName === 'TEXTAREA' ||
          target.tagName === 'SELECT' ||
          target.closest('button') ||
          target.getAttribute('role') === 'button' ||
          target.classList.contains('clickable') ||
          target.dataset.cursor === 'pointer';
        setIsPointer(!!isClickable);
      }
    };

    const handleMouseDown = () => setIsDown(true);
    const handleMouseUp = () => setIsDown(false);
    const handleMouseLeave = () => setIsVisible(false);

    window.addEventListener('mousemove', handleMouseMove, { passive: true });
    window.addEventListener('mousedown', handleMouseDown);
    window.addEventListener('mouseup', handleMouseUp);
    document.addEventListener('mouseleave', handleMouseLeave);

    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mousedown', handleMouseDown);
      window.removeEventListener('mouseup', handleMouseUp);
      document.removeEventListener('mouseleave', handleMouseLeave);
    };
  }, []);

  // Frame loop for spring-damped lagging reticle
  useEffect(() => {
    let animId: number;
    const updateTrail = () => {
      setTrail((prev) => ({
        x: prev.x + (pos.x - prev.x) * 0.28,
        y: prev.y + (pos.y - prev.y) * 0.28,
      }));
      animId = requestAnimationFrame(updateTrail);
    };
    animId = requestAnimationFrame(updateTrail);
    return () => cancelAnimationFrame(animId);
  }, [pos]);

  if (!isVisible) return null;

  return (
    <>
      {/* Lagging Technical Reticle */}
      <div
        style={{
          position: 'fixed',
          top: trail.y - 14,
          left: trail.x - 14,
          width: 28,
          height: 28,
          border: `1px solid ${isPointer ? '#ff4d1a' : 'rgba(236, 231, 223, 0.3)'}`,
          pointerEvents: 'none',
          zIndex: 9999,
          transform: `scale(${isDown ? 0.8 : isPointer ? 1.4 : 1}) rotate(${isPointer ? '45deg' : '0deg'})`,
          transition:
            'transform 0.22s cubic-bezier(0.16, 1, 0.3, 1), border-color 0.22s cubic-bezier(0.16, 1, 0.3, 1)',
        }}
      >
        {/* 4 Cardinal Hairline Crosshair Ticks */}
        <div
          style={{
            position: 'absolute',
            top: -4,
            left: 13,
            width: 1,
            height: 4,
            background: isPointer ? '#ff4d1a' : 'rgba(236, 231, 223, 0.4)',
          }}
        />
        <div
          style={{
            position: 'absolute',
            bottom: -4,
            left: 13,
            width: 1,
            height: 4,
            background: isPointer ? '#ff4d1a' : 'rgba(236, 231, 223, 0.4)',
          }}
        />
        <div
          style={{
            position: 'absolute',
            left: -4,
            top: 13,
            height: 1,
            width: 4,
            background: isPointer ? '#ff4d1a' : 'rgba(236, 231, 223, 0.4)',
          }}
        />
        <div
          style={{
            position: 'absolute',
            right: -4,
            top: 13,
            height: 1,
            width: 4,
            background: isPointer ? '#ff4d1a' : 'rgba(236, 231, 223, 0.4)',
          }}
        />
      </div>

      {/* Immediate Precision Center Pip */}
      <div
        style={{
          position: 'fixed',
          top: pos.y - 2,
          left: pos.x - 2,
          width: 4,
          height: 4,
          borderRadius: '50%',
          backgroundColor: '#ff4d1a',
          pointerEvents: 'none',
          zIndex: 10000,
        }}
      />
    </>
  );
};

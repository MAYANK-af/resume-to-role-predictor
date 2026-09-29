import React, { useRef, useEffect } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';
import gsap from 'gsap';
import type { RoleName } from '../types';
import { ANCHORS_CONFIG } from './RoleAnchors';

interface CameraRigProps {
  state: string;
  predictedRole: RoleName | null;
}

export const CameraRig: React.FC<CameraRigProps> = ({ state, predictedRole }) => {
  const { camera } = useThree();
  const mouseRef = useRef({ x: 0, y: 0 });
  const targetCamPos = useRef(new THREE.Vector3(0, 1.6, 12.8));
  const lookAtTarget = useRef(new THREE.Vector3(0, 1.6, 0));

  // Mouse Parallax Listener
  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      mouseRef.current.x = (e.clientX / window.innerWidth - 0.5) * 2;
      mouseRef.current.y = -(e.clientY / window.innerHeight - 0.5) * 2;
    };

    window.addEventListener('mousemove', handleMouseMove, { passive: true });
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, []);

  // Dolly on state changes with GSAP
  useEffect(() => {
    let toX = 0;
    let toY = 1.6;
    let toZ = 12.8;
    let lookX = 0;
    let lookY = 1.6;
    let lookZ = 0;

    if (state === 'landing') {
      toX = 0;
      toY = 1.6;
      toZ = 12.8;
      lookX = 0;
      lookY = 1.6;
    } else if (state === 'typing') {
      toX = 0;
      toY = 1.4;
      toZ = 11.8;
      lookX = 0;
      lookY = 1.4;
    } else if (state === 'analyzing') {
      const anchor = ANCHORS_CONFIG.find((a) => a.role === predictedRole);
      if (anchor) {
        toX = anchor.position[0] * 0.45;
        toY = anchor.position[1] * 0.45 + 0.4;
        toZ = 9.8;
        lookX = anchor.position[0] * 0.35;
        lookY = anchor.position[1] * 0.35;
      } else {
        toX = 0;
        toY = 1.8;
        toZ = 10.5;
        lookX = 0;
        lookY = 1.8;
      }
    } else if (state === 'result') {
      toX = 0;
      toY = 1.8;
      toZ = 11.2;
      lookX = 0;
      lookY = 1.6;
    }

    gsap.to(targetCamPos.current, {
      x: toX,
      y: toY,
      z: toZ,
      duration: 2.2,
      ease: 'expo.out',
    });

    gsap.to(lookAtTarget.current, {
      x: lookX,
      y: lookY,
      z: lookZ,
      duration: 2.2,
      ease: 'expo.out',
    });
  }, [state, predictedRole]);

  useFrame((_, delta) => {
    // Subtle mouse parallax
    const parallaxX = mouseRef.current.x * 0.35;
    const parallaxY = mouseRef.current.y * 0.25;

    camera.position.x = THREE.MathUtils.damp(
      camera.position.x,
      targetCamPos.current.x + parallaxX,
      2.5,
      delta
    );
    camera.position.y = THREE.MathUtils.damp(
      camera.position.y,
      targetCamPos.current.y + parallaxY,
      2.5,
      delta
    );
    camera.position.z = THREE.MathUtils.damp(
      camera.position.z,
      targetCamPos.current.z,
      2.5,
      delta
    );

    camera.lookAt(lookAtTarget.current);
  });

  return null;
};

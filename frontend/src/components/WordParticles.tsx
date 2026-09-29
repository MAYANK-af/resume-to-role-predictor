import React, { useRef, useMemo, useEffect, useState } from 'react';
import { useFrame } from '@react-three/fiber';
import { Text, Line } from '@react-three/drei';
import * as THREE from 'three';
import type { WordWeight, RoleName } from '../types';
import { ANCHORS_CONFIG } from './RoleAnchors';

interface WordParticlesProps {
  resumeText: string;
  state: string;
  predictedRole: RoleName | null;
  topWords: WordWeight[];
  onHoverWord: (word: WordWeight | null) => void;
}

interface DynamicWordNode {
  id: string;
  word: string;
  initPos: THREE.Vector3;
  targetPos: THREE.Vector3;
  weight: number;
  isInfluential: boolean;
}

export const WordParticles: React.FC<WordParticlesProps> = ({
  resumeText,
  state,
  predictedRole,
  topWords,
  onHoverWord,
}) => {
  // 1. Ambient Background Drifting Dust Particles
  const ambientPointsRef = useRef<THREE.Points>(null);
  const particleCount = 750;

  const [ambientPositions] = useMemo(() => {
    const pos = new Float32Array(particleCount * 3);
    for (let i = 0; i < particleCount * 3; i += 3) {
      pos[i] = (Math.random() - 0.5) * 26;
      pos[i + 1] = (Math.random() - 0.5) * 18;
      pos[i + 2] = (Math.random() - 0.5) * 14;
    }
    return [pos];
  }, []);

  useFrame((_, delta) => {
    if (ambientPointsRef.current) {
      ambientPointsRef.current.rotation.y += delta * 0.015;
      ambientPointsRef.current.rotation.x += delta * 0.008;
    }
  });

  // 2. Dynamic Resume Words
  const [nodes, setNodes] = useState<DynamicWordNode[]>([]);

  const targetAnchor = useMemo(
    () => ANCHORS_CONFIG.find((a) => a.role === predictedRole),
    [predictedRole]
  );
  const anchorPos = useMemo(
    () => (targetAnchor ? new THREE.Vector3(...targetAnchor.position) : new THREE.Vector3(0, 0, 0)),
    [targetAnchor]
  );

  useEffect(() => {
    if (!resumeText.trim()) {
      setNodes([]);
      return;
    }

    // Extract unique words from text
    const rawTokens = resumeText
      .toLowerCase()
      .replace(/[^a-z0-9+#\.\s]/g, ' ')
      .split(/\s+/)
      .filter((w) => w.length > 2);

    const uniqueWords = Array.from(new Set(rawTokens)).slice(0, 36);

    const topWordMap = new Map<string, WordWeight>();
    topWords.forEach((tw) => {
      topWordMap.set(tw.word.toLowerCase(), tw);
    });

    const newNodes: DynamicWordNode[] = uniqueWords.map((word, idx) => {
      const match = topWordMap.get(word);
      const isInfluential = !!match;
      const weight = match ? match.weight : 0.04;

      // Initial ambient sphere spread
      const radius = 3.6 + Math.random() * 2.4;
      const theta = (idx / uniqueWords.length) * Math.PI * 2;
      const phi = (Math.random() - 0.5) * Math.PI * 0.8;

      const initX = radius * Math.cos(theta) * Math.cos(phi);
      const initY = 1.0 + radius * Math.sin(phi);
      const initZ = radius * Math.sin(theta) * Math.cos(phi);

      const initPos = new THREE.Vector3(initX, initY, initZ);
      let targetPos = initPos.clone();

      if ((state === 'analyzing' || state === 'result') && targetAnchor) {
        if (isInfluential) {
          // Cluster around predicted anchor
          const spread = 1.6;
          const jitter = new THREE.Vector3(
            (Math.random() - 0.5) * spread,
            (Math.random() - 0.5) * spread,
            (Math.random() - 0.5) * (spread * 0.8)
          );
          targetPos = anchorPos.clone().add(jitter);
        } else {
          // Drift outward into deep background space
          targetPos = initPos.clone().multiplyScalar(1.6);
        }
      }

      return {
        id: `${word}-${idx}`,
        word,
        initPos,
        targetPos,
        weight,
        isInfluential,
      };
    });

    setNodes(newNodes);
  }, [resumeText, state, predictedRole, topWords, targetAnchor, anchorPos]);

  return (
    <group name="word-constellation">
      {/* Ambient background dust particles */}
      <points ref={ambientPointsRef}>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            args={[ambientPositions, 3]}
          />
        </bufferGeometry>
        <pointsMaterial
          size={0.09}
          color="#ece7df"
          transparent={true}
          opacity={0.3}
          sizeAttenuation={true}
        />
      </points>

      {/* Constellation Connector Lines */}
      {(state === 'analyzing' || state === 'result') &&
        targetAnchor &&
        nodes
          .filter((n) => n.isInfluential)
          .map((n) => (
            <Line
              key={`line-${n.id}`}
              points={[n.targetPos, anchorPos]}
              color="#ff4d1a"
              transparent={true}
              opacity={0.3}
              lineWidth={1}
            />
          ))}

      {/* Dynamic 3D Word Nodes */}
      {nodes.map((node) => (
        <SingleWordItem
          key={node.id}
          node={node}
          state={state}
          onHover={onHoverWord}
        />
      ))}
    </group>
  );
};

interface SingleWordItemProps {
  node: DynamicWordNode;
  state: string;
  onHover: (word: WordWeight | null) => void;
}

const SingleWordItem: React.FC<SingleWordItemProps> = ({ node, state, onHover }) => {
  const groupRef = useRef<THREE.Group>(null);
  const currentPos = useRef(node.initPos.clone());

  const isTargeted = node.isInfluential && (state === 'analyzing' || state === 'result');
  const color = isTargeted ? '#ff4d1a' : '#ece7df';
  const opacity = isTargeted ? 1.0 : state === 'result' ? 0.12 : 0.65;
  const fontSize = node.isInfluential
    ? Math.min(0.24 + node.weight * 0.55, 0.56)
    : 0.20;

  useFrame((_, delta) => {
    if (!groupRef.current) return;

    // Speed scales with weight for influential words
    const baseSpeed = state === 'analyzing' ? 3.5 : 1.6;
    const lerpSpeed = node.isInfluential ? baseSpeed + node.weight * 6.0 : 1.0;

    currentPos.current.lerp(node.targetPos, delta * lerpSpeed);
    groupRef.current.position.copy(currentPos.current);
  });

  return (
    <group ref={groupRef} position={node.initPos}>
      <Text
        fontSize={fontSize}
        color={color}
        fillOpacity={opacity}
        anchorX="center"
        anchorY="middle"
        letterSpacing={0.08}
        onPointerOver={(e) => {
          e.stopPropagation();
          onHover({ word: node.word, weight: node.weight });
        }}
        onPointerOut={() => onHover(null)}
      >
        {node.word}
      </Text>
    </group>
  );
};

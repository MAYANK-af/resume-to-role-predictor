import React, { useRef, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import { Text, Html } from '@react-three/drei';
import * as THREE from 'three';
import type { RoleName, RoleAnchorInfo } from '../types';

interface RoleAnchorsProps {
  predictedRole: RoleName | null;
  hoveredRole: RoleName | null;
  onHoverRole: (role: RoleName | null) => void;
  state: string;
  topCoefficients?: Record<RoleName, Array<{ display: string; weight: number }>>;
}

export const ANCHORS_CONFIG: RoleAnchorInfo[] = [
  {
    role: 'Backend',
    name: 'BACKEND',
    position: [-5.2, 2.7, -1.4],
    geometryType: 'box',
    description: 'Java / Python / DotNet / Database / DevOps',
  },
  {
    role: 'AI/ML',
    name: 'AI / ML',
    position: [-2.2, 2.0, -0.6],
    geometryType: 'icosahedron',
    description: 'Data Science / Deep Learning / Predictive Modeling',
  },
  {
    role: 'Other Tech',
    name: 'OTHER TECH',
    position: [0.0, 3.4, -1.2],
    geometryType: 'torus',
    description: 'QA / Automation Testing / Security / Blockchain',
  },
  {
    role: 'Web Dev',
    name: 'WEB DEV',
    position: [2.2, 2.0, -0.6],
    geometryType: 'octahedron',
    description: 'HTML5 / CSS3 / JavaScript / UI Systems',
  },
  {
    role: 'Analyst',
    name: 'ANALYST',
    position: [5.2, 2.7, -1.4],
    geometryType: 'dodecahedron',
    description: 'Business Intelligence / KPI / Requirement Gathering',
  },
];

export const RoleAnchors: React.FC<RoleAnchorsProps> = ({
  predictedRole,
  hoveredRole,
  onHoverRole,
  state,
  topCoefficients,
}) => {
  return (
    <group name="role-anchors">
      {ANCHORS_CONFIG.map((anchor) => (
        <SingleAnchor
          key={anchor.role}
          anchor={anchor}
          isPredicted={predictedRole === anchor.role}
          isHovered={hoveredRole === anchor.role}
          hasPrediction={Boolean(predictedRole && (state === 'analyzing' || state === 'result'))}
          onHover={(hover) => onHoverRole(hover ? anchor.role : null)}
          topWords={topCoefficients ? topCoefficients[anchor.role] : []}
        />
      ))}
    </group>
  );
};

interface SingleAnchorProps {
  anchor: RoleAnchorInfo;
  isPredicted: boolean;
  isHovered: boolean;
  hasPrediction: boolean;
  onHover: (hover: boolean) => void;
  topWords?: Array<{ display: string; weight: number }>;
}

const SingleAnchor: React.FC<SingleAnchorProps> = ({
  anchor,
  isPredicted,
  isHovered,
  hasPrediction,
  onHover,
  topWords = [],
}) => {
  const groupRef = useRef<THREE.Group>(null);
  const coreRef = useRef<THREE.Mesh>(null);
  const ringRef = useRef<THREE.LineSegments>(null);

  // Styling based on state: when prediction is active, non-predicted anchors fade down
  let strokeColor = '#ece7df';
  let edgeOpacity = 0.45;
  let coreOpacity = 0.12;
  let labelOpacity = 0.7;
  let scale = 1.0;

  if (hasPrediction) {
    if (isPredicted) {
      strokeColor = '#ff4d1a';
      edgeOpacity = 1.0;
      coreOpacity = 0.35;
      labelOpacity = 1.0;
      scale = 1.18;
    } else {
      strokeColor = '#504c45';
      edgeOpacity = 0.16;
      coreOpacity = 0.04;
      labelOpacity = 0.22;
      scale = 0.9;
    }
  } else if (isHovered) {
    strokeColor = '#ff4d1a';
    edgeOpacity = 0.95;
    coreOpacity = 0.25;
    labelOpacity = 1.0;
    scale = 1.1;
  }

  useFrame((_, delta) => {
    if (groupRef.current) {
      const rotSpeed = isPredicted ? 0.75 : isHovered ? 0.6 : 0.2;
      groupRef.current.rotation.x += delta * rotSpeed * 0.6;
      groupRef.current.rotation.y += delta * rotSpeed;
    }
    if (ringRef.current) {
      ringRef.current.rotation.z -= delta * (isPredicted ? 0.5 : 0.2);
    }
  });

  const geometry = useMemo(() => {
    switch (anchor.geometryType) {
      case 'box':
        return new THREE.BoxGeometry(1.5, 1.5, 1.5);
      case 'icosahedron':
        return new THREE.IcosahedronGeometry(1.15, 0);
      case 'octahedron':
        return new THREE.OctahedronGeometry(1.2, 0);
      case 'dodecahedron':
        return new THREE.DodecahedronGeometry(1.1, 0);
      case 'torus':
        return new THREE.TorusGeometry(0.95, 0.3, 10, 26);
      default:
        return new THREE.BoxGeometry(1.3, 1.3, 1.3);
    }
  }, [anchor.geometryType]);

  const edges = useMemo(() => new THREE.EdgesGeometry(geometry), [geometry]);

  return (
    <group position={anchor.position} scale={scale}>
      {/* Rotating Geometric Anchor Group */}
      <group
        ref={groupRef}
        onPointerOver={(e) => {
          e.stopPropagation();
          onHover(true);
        }}
        onPointerOut={() => onHover(false)}
      >
        {/* Solid Matte Dark Core (not glossy) */}
        <mesh ref={coreRef} geometry={geometry}>
          <meshStandardMaterial
            color="#141210"
            roughness={0.92}
            metalness={0.08}
            transparent={true}
            opacity={coreOpacity}
          />
        </mesh>

        {/* Crisp Architectural Wireframe Edges */}
        <lineSegments geometry={edges}>
          <lineBasicMaterial
            color={strokeColor}
            transparent={true}
            opacity={edgeOpacity}
          />
        </lineSegments>

        {/* Outer Orbiting Tick Halo */}
        <lineSegments ref={ringRef} geometry={edges} scale={1.22}>
          <lineBasicMaterial
            color={isPredicted ? '#ff4d1a' : '#ece7df'}
            transparent={true}
            opacity={isPredicted ? 0.6 : hasPrediction ? 0.08 : 0.16}
          />
        </lineSegments>
      </group>

      {/* 3D Discipline Label without // prefix */}
      <Text
        position={[0, -1.35, 0]}
        fontSize={0.22}
        color={strokeColor}
        fillOpacity={labelOpacity}
        anchorX="center"
        anchorY="top"
        letterSpacing={0.12}
      >
        {anchor.name}
      </Text>

      {/* Hover Inspection HUD Overlay without brackets */}
      {isHovered && (
        <Html distanceFactor={13} position={[0, 1.6, 0]} center>
          <div
            style={{
              background: '#141210',
              border: '1px solid #ff4d1a',
              color: '#ece7df',
              fontFamily: "'JetBrains Mono', monospace",
              fontSize: '11px',
              padding: '12px 16px',
              whiteSpace: 'nowrap',
              pointerEvents: 'none',
              boxShadow: '0 12px 32px rgba(0,0,0,0.9)',
              letterSpacing: '0.06em',
              zIndex: 1000,
            }}
          >
            <div style={{ color: '#ff4d1a', fontWeight: 600, marginBottom: '4px' }}>
              {anchor.name}
            </div>
            <div style={{ color: '#8a847b', fontSize: '10px', marginBottom: '8px' }}>
              {anchor.description}
            </div>
            {topWords.length > 0 && (
              <div style={{ borderTop: '1px solid #262320', paddingTop: '6px' }}>
                <span style={{ color: '#8a847b' }}>Key skills: </span>
                <span style={{ color: '#ece7df', fontWeight: 500 }}>
                  {topWords.slice(0, 4).map((w) => w.display).join(', ')}
                </span>
              </div>
            )}
          </div>
        </Html>
      )}
    </group>
  );
};

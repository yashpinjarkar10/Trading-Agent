import { Suspense, useMemo, useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { Float, OrbitControls } from '@react-three/drei';
import * as THREE from 'three';

/**
 * HeroScene
 * A subtle, performance-friendly 3D scene: a field of animated "candlesticks"
 * over a grid plane with floating glow orbs. Used in empty-state panels and
 * welcome screens to give the product a premium "AI-native" feel.
 */
export default function HeroScene({ className }) {
  return (
    <div className={className} style={{ position: 'absolute', inset: 0 }}>
      <Canvas
        camera={{ position: [0, 4.5, 9], fov: 45 }}
        dpr={[1, 1.6]}
        gl={{ antialias: true, alpha: true, powerPreference: 'high-performance' }}
      >
        <color attach="background" args={[0, 0, 0]} />
        <fog attach="fog" args={['#05060c', 8, 22]} />
        <ambientLight intensity={0.35} />
        <directionalLight position={[5, 8, 5]} intensity={0.6} color="#22d3ee" />
        <directionalLight position={[-5, 4, -3]} intensity={0.4} color="#8b5cf6" />
        <Suspense fallback={null}>
          <Candlesticks count={32} />
          <GridFloor />
          <FloatingOrbs />
        </Suspense>
        <OrbitControls
          enableZoom={false}
          enablePan={false}
          autoRotate
          autoRotateSpeed={0.4}
          minPolarAngle={Math.PI / 3}
          maxPolarAngle={Math.PI / 2.2}
        />
      </Canvas>
    </div>
  );
}

function Candlesticks({ count = 28 }) {
  const meshRef = useRef();
  const dummy = useMemo(() => new THREE.Object3D(), []);
  const color = useMemo(() => new THREE.Color(), []);
  const data = useMemo(() => {
    const arr = [];
    for (let i = 0; i < count; i++) {
      arr.push({
        x: (i - count / 2) * 0.45,
        baseHeight: 0.5 + Math.random() * 1.8,
        speed: 0.4 + Math.random() * 0.8,
        phase: Math.random() * Math.PI * 2,
        bullish: Math.random() > 0.5,
      });
    }
    return arr;
  }, [count]);

  useFrame((state) => {
    if (!meshRef.current) return;
    const t = state.clock.elapsedTime;
    for (let i = 0; i < data.length; i++) {
      const d = data[i];
      const h = d.baseHeight + Math.sin(t * d.speed + d.phase) * 0.6;
      dummy.position.set(d.x, h / 2, Math.sin(t * 0.2 + i) * 0.4);
      dummy.scale.set(0.18, h, 0.18);
      dummy.updateMatrix();
      meshRef.current.setMatrixAt(i, dummy.matrix);
      const upBias = Math.sin(t * d.speed + d.phase);
      const isUp = d.bullish ? upBias > -0.2 : upBias > 0.4;
      color.set(isUp ? '#22d3ee' : '#8b5cf6');
      meshRef.current.setColorAt(i, color);
    }
    meshRef.current.instanceMatrix.needsUpdate = true;
    if (meshRef.current.instanceColor) meshRef.current.instanceColor.needsUpdate = true;
  });

  return (
    <instancedMesh ref={meshRef} args={[null, null, count]} castShadow>
      <boxGeometry args={[1, 1, 1]} />
      <meshStandardMaterial
        metalness={0.4}
        roughness={0.25}
        emissiveIntensity={0.6}
        emissive="#0c1220"
        toneMapped={false}
      />
    </instancedMesh>
  );
}

function GridFloor() {
  return (
    <group position={[0, -0.01, 0]}>
      <gridHelper args={[40, 40, '#1a2030', '#11141f']} />
      <mesh rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[40, 40]} />
        <meshBasicMaterial color="#05060c" transparent opacity={0.6} />
      </mesh>
    </group>
  );
}

function FloatingOrbs() {
  const orbs = useMemo(
    () => [
      { pos: [-3.8, 2.6, -1.5], color: '#22d3ee', size: 0.18 },
      { pos: [3.2, 3.4, -0.8], color: '#8b5cf6', size: 0.22 },
      { pos: [0.8, 4.1, -2.2], color: '#22d3ee', size: 0.14 },
      { pos: [-2.0, 3.0, 1.5], color: '#8b5cf6', size: 0.16 },
    ],
    [],
  );
  return (
    <>
      {orbs.map((o, i) => (
        <Float key={i} speed={1.2 + i * 0.2} rotationIntensity={0} floatIntensity={1.4}>
          <mesh position={o.pos}>
            <sphereGeometry args={[o.size, 24, 24]} />
            <meshStandardMaterial
              color={o.color}
              emissive={o.color}
              emissiveIntensity={1.4}
              toneMapped={false}
            />
          </mesh>
        </Float>
      ))}
    </>
  );
}

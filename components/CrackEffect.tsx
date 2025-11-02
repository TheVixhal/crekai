'use client'

import { useState } from 'react'

export default function CrackEffect({ children }) {
  const [cracks, setCracks] = useState([])

  const handleClick = (e) => {
    const x = e.clientX
    const y = e.clientY
    const id = Date.now()
    
    // Random rotation for variety
    const rotation = Math.random() * 360
    // Random scale between 0.7 and 1.3
    const scale = 0.7 + Math.random() * 0.6
    
    const newCrack = {
      id,
      x,
      y,
      rotation,
      scale
    }
    
    setCracks(prev => [...prev, newCrack])
    
    // Remove crack after animation completes
    setTimeout(() => {
      setCracks(prev => prev.filter(crack => crack.id !== id))
    }, 3000)
  }

  return (
    <div onClick={handleClick} className="relative w-full h-full">
      {children}
      
      {/* Crack overlay */}
      <div className="fixed inset-0 pointer-events-none z-50">
        {cracks.map(crack => (
          <img 
            key={crack.id}
            src="/crack.png" 
            alt="" 
            className="absolute w-72 h-72 md:w-96 md:h-96"
            style={{ 
              left: `${crack.x}px`,
              top: `${crack.y}px`,
              transform: `translate(-50%, -50%) rotate(${crack.rotation}deg) scale(${crack.scale})`,
              filter: 'drop-shadow(0 2px 8px rgba(0,0,0,0.15)) brightness(0.4) contrast(1.2)',
              mixBlendMode: 'darken',
              opacity: 0,
              animation: 'crackAppear 0.4s ease-out forwards, crackFadeOut 1s ease-out forwards 1.8s'
            }}
          />
        ))}
      </div>

      <style jsx>{`
        @keyframes crackAppear {
          0% {
            opacity: 0;
            transform: translate(-50%, -50%) rotate(${0}deg) scale(0.1);
          }
          20% {
            opacity: 0.95;
            transform: translate(-50%, -50%) scale(0.5);
          }
          60% {
            opacity: 0.9;
          }
          100% {
            opacity: 0.85;
          }
        }
        
        @keyframes crackFadeOut {
          0% {
            opacity: 0.85;
          }
          100% {
            opacity: 0;
          }
        }
      `}</style>
    </div>
  )
}

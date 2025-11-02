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
    // Random scale between 0.8 and 1.5
    const scale = 0.8 + Math.random() * 0.7
    
    const newCrack = {
      id,
      x,
      y,
      rotation,
      scale
    }
    
    setCracks(prev => [...prev, newCrack])
    
    // Remove crack after animation
    setTimeout(() => {
      setCracks(prev => prev.filter(crack => crack.id !== id))
    }, 2000)
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
            className="absolute w-64 h-64 md:w-80 md:h-80"
            style={{ 
              left: `${crack.x}px`,
              top: `${crack.y}px`,
              transform: `translate(-50%, -50%) rotate(${crack.rotation}deg) scale(${crack.scale})`,
              filter: 'drop-shadow(0 0 10px rgba(0,0,0,0.3))',
              mixBlendMode: 'multiply',
              animation: 'crackAppear 0.15s cubic-bezier(0.34, 1.56, 0.64, 1) forwards, crackFadeOut 0.8s ease-out forwards 1.2s'
            }}
          />
        ))}
      </div>

      <style jsx>{`
        @keyframes crackAppear {
          0% {
            opacity: 0;
            transform: translate(-50%, -50%) rotate(0deg) scale(0.3);
          }
          100% {
            opacity: 1;
          }
        }
        
        @keyframes crackFadeOut {
          0% {
            opacity: 1;
          }
          100% {
            opacity: 0;
          }
        }
      `}</style>
    </div>
  )
}

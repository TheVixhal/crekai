'use client'

import { useState } from 'react'
import Image from 'next/image'

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
      <div className="fixed inset-0 pointer-events-none z-50 overflow-hidden">
        {cracks.map(crack => (
          <div
            key={crack.id}
            className="absolute"
            style={{
              left: crack.x,
              top: crack.y,
              transform: `translate(-50%, -50%) rotate(${crack.rotation}deg) scale(${crack.scale})`,
              animation: 'crackAppear 0.2s ease-out forwards, crackFadeOut 0.8s ease-out forwards 1s'
            }}
          >
            <img 
              src="/crack.png" 
              alt="" 
              className="w-64 h-64 md:w-80 md:h-80"
              style={{ 
                filter: 'drop-shadow(0 0 10px rgba(0,0,0,0.3))',
                mixBlendMode: 'multiply'
              }}
            />
          </div>
        ))}
      </div>

      <style jsx>{`
        @keyframes crackAppear {
          0% {
            opacity: 0;
            transform: translate(-50%, -50%) rotate(${0}deg) scale(0.5);
          }
          100% {
            opacity: 1;
            transform: translate(-50%, -50%) rotate(var(--rotation)) scale(var(--scale));
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

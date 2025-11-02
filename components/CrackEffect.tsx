'use client'

import { useState } from 'react'

export default function CrackEffect({ children }) {
  const [cracks, setCracks] = useState([])

  const handleClick = (e) => {
    const x = e.clientX
    const y = e.clientY
    const id = Date.now()
    
    // Random scale between 0.7 and 1.3
    const scale = 0.7 + Math.random() * 0.6
    
    // Calculate position to center the image (assuming 384px = 96 * 4 for w-96)
    const imageSize = 384 // md:w-96 = 24rem = 384px
    const centeredX = x - (imageSize / 2)
    const centeredY = y - (imageSize / 2)
    
    const newCrack = {
      id,
      x: centeredX,
      y: centeredY,
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
            src="/crack1.png" 
            alt="" 
            className="absolute w-96 h-96"
            style={{ 
              left: `${crack.x}px`,
              top: `${crack.y}px`,
              transform: `scale(${crack.scale})`,
              transformOrigin: 'center center',
              filter: 'brightness(0) contrast(2) drop-shadow(0 2px 4px rgba(0,0,0,0.3))',
              mixBlendMode: 'multiply',
              opacity: 0,
              animation: 'crackAppear 0.1s ease-out forwards, crackFadeOut 1s ease-out forwards 1.5s'
            }}
          />
        ))}
      </div>

      <style jsx>{`
        @keyframes crackAppear {
          0% {
            opacity: 0;
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

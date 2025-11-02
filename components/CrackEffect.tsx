'use client'

import { useState, useEffect } from 'react'

export default function CrackEffect({ children }) {
  const [cracks, setCracks] = useState([])

  const generateCrack = (x, y) => {
    const crackLines = []
    const numLines = 8 + Math.floor(Math.random() * 5) // 8-12 crack lines
    
    for (let i = 0; i < numLines; i++) {
      const angle = (Math.PI * 2 * i) / numLines + (Math.random() - 0.5) * 0.5
      const length = 30 + Math.random() * 70 // Random length between 30-100px
      const endX = x + Math.cos(angle) * length
      const endY = y + Math.sin(angle) * length
      
      // Add some randomness to make it look more natural
      const midX = x + Math.cos(angle) * (length / 2) + (Math.random() - 0.5) * 20
      const midY = y + Math.sin(angle) * (length / 2) + (Math.random() - 0.5) * 20
      
      crackLines.push({
        path: `M ${x} ${y} Q ${midX} ${midY} ${endX} ${endY}`,
        delay: Math.random() * 100
      })
    }
    
    return crackLines
  }

  const handleClick = (e) => {
    const x = e.clientX
    const y = e.clientY
    const id = Date.now()
    
    const newCrack = {
      id,
      x,
      y,
      lines: generateCrack(x, y)
    }
    
    setCracks(prev => [...prev, newCrack])
    
    // Remove crack after animation
    setTimeout(() => {
      setCracks(prev => prev.filter(crack => crack.id !== id))
    }, 1500)
  }

  return (
    <div onClick={handleClick} className="relative w-full h-full">
      {children}
      
      {/* Crack overlay */}
      <svg 
        className="fixed inset-0 pointer-events-none z-50"
        style={{ width: '100vw', height: '100vh' }}
      >
        {cracks.map(crack => (
          <g key={crack.id}>
            {crack.lines.map((line, index) => (
              <path
                key={index}
                d={line.path}
                stroke="rgba(0, 0, 0, 0.6)"
                strokeWidth="2"
                fill="none"
                strokeLinecap="round"
                style={{
                  animation: `crackAppear 0.3s ease-out forwards ${line.delay}ms, crackFade 0.5s ease-out forwards 0.8s`,
                  strokeDasharray: '100',
                  strokeDashoffset: '100'
                }}
              />
            ))}
            {/* Impact circle */}
            <circle
              cx={crack.x}
              cy={crack.y}
              r="0"
              fill="rgba(0, 0, 0, 0.1)"
              style={{
                animation: 'impactPulse 0.4s ease-out forwards'
              }}
            />
          </g>
        ))}
      </svg>

      <style jsx>{`
        @keyframes crackAppear {
          to {
            strokeDashoffset: 0;
          }
        }
        
        @keyframes crackFade {
          to {
            opacity: 0;
          }
        }
        
        @keyframes impactPulse {
          0% {
            r: 0;
            opacity: 0.4;
          }
          50% {
            r: 20;
            opacity: 0.2;
          }
          100% {
            r: 30;
            opacity: 0;
          }
        }
      `}</style>
    </div>
  )
}

'use client'

import { useState, useEffect } from 'react'

export default function TypewriterHeading({ 
  texts = [
    "Build Artificial Neural Network From Scratch",
    "Build Your Own ChatGPT",
  ],
  typingSpeed = 80,
  deletingSpeed = 40,
  pauseAfterComplete = 2000,
  pauseBeforeDelete = 1500
}) {
  const [textIndex, setTextIndex] = useState(0)
  const [charIndex, setCharIndex] = useState(0)
  const [isDeleting, setIsDeleting] = useState(false)
  const [displayText, setDisplayText] = useState('')

  useEffect(() => {
    const currentFullText = texts[textIndex]
    
    const handleTyping = () => {
      if (!isDeleting) {
        // Typing forward
        if (charIndex < currentFullText.length) {
          setDisplayText(currentFullText.substring(0, charIndex + 1))
          setCharIndex(charIndex + 1)
        } else {
          // Finished typing, wait then start deleting
          setTimeout(() => setIsDeleting(true), pauseBeforeDelete)
        }
      } else {
        // Deleting backward
        if (charIndex > 0) {
          setDisplayText(currentFullText.substring(0, charIndex - 1))
          setCharIndex(charIndex - 1)
        } else {
          // Finished deleting, move to next text
          setIsDeleting(false)
          setTextIndex((textIndex + 1) % texts.length)
        }
      }
    }

    const timeout = setTimeout(
      handleTyping,
      isDeleting ? deletingSpeed : typingSpeed
    )

    return () => clearTimeout(timeout)
  }, [charIndex, isDeleting, textIndex, texts, typingSpeed, deletingSpeed, pauseBeforeDelete])

  return (
    <h2 className="text-4xl md:text-5xl font-bold mb-3 text-black min-h-[3rem] md:min-h-[4rem] flex items-center">
      <span>{displayText}</span>
      <span className="ml-1 inline-block w-0.5 h-10 md:h-12 bg-black animate-blink"></span>
      <style jsx>{`
        @keyframes blink {
          0%, 50% { opacity: 1; }
          51%, 100% { opacity: 0; }
        }
        .animate-blink {
          animation: blink 1s infinite;
        }
      `}</style>
    </h2>
  )
}

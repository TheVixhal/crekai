"use client"

import Image from "next/image"

interface WelcomeBannerProps {
  fullName: string
}

export default function WelcomeBanner({ fullName }: WelcomeBannerProps) {
  return (
    <>
      <style jsx>{`
        @keyframes cloudFloat {
          0% {
            transform: translateX(-100px) translateY(0px);
          }
          50% {
            transform: translateX(100px) translateY(-15px);
          }
          100% {
            transform: translateX(-100px) translateY(0px);
          }
        }
        .cloud-float {
          animation: cloudFloat 20s ease-in-out infinite;
        }
      `}</style>

      <div className="bg-gradient-to-r from-cyan-400 to-cyan-600 border-4 border-black p-8 shadow-[8px_8px_0px_0px_rgba(0,0,0,1)] relative">
        {/* Floating Cloud - positioned above the banner */}
        <div className="absolute -top-16 left-1/4 cloud-float z-20 pointer-events-none">
          <Image
            src="/cloud.png"
            alt="Floating Cloud"
            width={100}
            height={70}
            className="drop-shadow-2xl opacity-90"
            priority
          />
        </div>
        
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-3xl font-bold text-white font-serif mb-2 drop-shadow-lg">
              Welcome Back, {fullName}!
            </h2>
            <p className="text-white/90 font-sans text-lg">
              Continue your AI/ML learning journey
            </p>
          </div>
          <div className="hidden md:block text-6xl">🚀</div>
        </div>
      </div>
    </>
  )
}

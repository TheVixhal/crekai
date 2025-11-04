"use client"

import Link from "next/link"

interface ProjectCardProps {
  project: {
    id: string
    slug: string
    title: string
    description: string
    total_steps: number
  }
  progress?: {
    current_step: number
    completed_steps: number[]
  }
}

export default function ProjectCard({ project, progress }: ProjectCardProps) {
  const currentStep = progress?.current_step || 1
  const completedSteps = progress?.completed_steps?.length || 0
  const progressPercent = (completedSteps / project.total_steps) * 100

  return (
    <div className="bg-white border-2 border-black hover:shadow-lg transition group">
      <div className="border-b-2 border-black p-4 bg-gray-100">
        <h3 className="text-xl font-bold text-black font-serif">{project.title}</h3>
      </div>

      <div className="p-6">
        <p className="text-gray-700 text-sm mb-4 font-sans">{project.description}</p>

        {progress && (
          <div className="mb-6">
            <div className="flex justify-between text-xs font-bold mb-2 font-sans">
              <span className="text-black">PROGRESS</span>
              <span className="text-black">{Math.round(progressPercent)}%</span>
            </div>
            <div className="w-full bg-gray-300 h-2 border border-black">
              <div className="bg-black h-full transition-all" style={{ width: `${progressPercent}%` }} />
            </div>
          </div>
        )}

        <Link href={`/projects/${project.slug}`}>
          <button className="w-full bg-black text-white font-serif font-bold py-2 border-2 border-black hover:bg-gray-900 transition">
            {progress ? `Continue (Step ${currentStep})` : "Begin Project"}
          </button>
        </Link>
      </div>
    </div>
  )
}

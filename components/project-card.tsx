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
    <div className="group bg-white/60 backdrop-blur-sm border border-gray-200 rounded-xl shadow-sm hover:shadow-md transition-all duration-200">
      <div className="border-b border-gray-100 px-5 py-4 bg-gradient-to-r from-gray-50 to-gray-100 rounded-t-xl">
        <h3 className="text-lg font-semibold text-gray-900 tracking-tight">{project.title}</h3>
      </div>

      <div className="p-6">
        <p className="text-gray-600 text-sm leading-relaxed mb-5">
          {project.description}
        </p>

        {progress && (
          <div className="mb-6">
            <div className="flex justify-between text-xs font-medium mb-2 text-gray-700">
              <span>Progress</span>
              <span>{Math.round(progressPercent)}%</span>
            </div>
            <div className="w-full bg-gray-200/70 rounded-full h-2 overflow-hidden">
              <div
                className="bg-gradient-to-r from-indigo-500 to-blue-400 h-full transition-all duration-500 rounded-full"
                style={{ width: `${progressPercent}%` }}
              />
            </div>
          </div>
        )}

        <Link href={`/projects/${project.slug}`}>
          <button className="w-full bg-gray-900 text-white text-sm font-medium py-2.5 rounded-md hover:bg-gray-800 transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-gray-900">
            {progress ? `Continue (Step ${currentStep})` : "Begin Project"}
          </button>
        </Link>
      </div>
    </div>
  )
}

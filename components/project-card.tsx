"use client"

import Link from "next/link"
import { useRouter } from "next/navigation"

interface ProjectCardProps {
  project: {
    id: string
    slug: string
    title: string
    description: string
    total_steps: number
    isFree?: boolean
  }
  progress?: {
    current_step: number
    completed_steps: number[]
  }
  userLevel: number
}

export default function ProjectCard({ project, progress, userLevel }: ProjectCardProps) {
  const router = useRouter()
  const completedSteps = progress?.completed_steps?.length || 0
  const totalSteps = project.total_steps
  const isFree = project.isFree ?? true
  const isLocked = !isFree && userLevel === 0

  const handleClick = (e: React.MouseEvent) => {
    if (isLocked) {
      e.preventDefault()
      router.push('/subscription')
    }
  }

  return (
    <Link 
      href={isLocked ? '#' : `/projects/${project.slug}`} 
      className="block group"
      onClick={handleClick}
    >
      <div className={`bg-white rounded-xl border border-gray-200 hover:border-gray-300 hover:shadow-md transition-all p-6 h-full flex flex-col justify-between ${isLocked ? 'opacity-75' : ''}`}>
        <div>
          <div className="flex items-start justify-between mb-3">
            <h3 className="text-xl font-semibold text-gray-900 group-hover:text-gray-700 transition-colors pr-4">
              {project.title}
            </h3>
            <div className="flex-shrink-0 w-10 h-10 bg-gray-100 rounded-lg flex items-center justify-center">
              {isLocked ? (
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <rect x="3" y="11" width="18" height="11" rx="2" ry="2"></rect>
                  <path d="M7 11V7a5 5 0 0 1 10 0v4"></path>
                </svg>
              ) : (
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <polyline points="9 18 15 12 9 6"></polyline>
                </svg>
              )}
            </div>
          </div>
          
          <p className="text-gray-600 text-sm leading-relaxed mb-4">
            {project.description}
          </p>
        </div>

        <div className="flex items-center justify-between pt-4 border-t border-gray-100">
          {progress ? (
            <>
              <div className="flex items-center gap-2 text-sm text-gray-500">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M12 2v20M2 12h20"></path>
                </svg>
                <span>{completedSteps}/{totalSteps} steps</span>
              </div>
              <span className="text-sm font-medium text-gray-900">
                {Math.round((completedSteps / totalSteps) * 100)}%
              </span>
            </>
          ) : (
            <>
              <div className="flex items-center gap-2 text-sm text-gray-500">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M12 2v20M2 12h20"></path>
                </svg>
                <span>{totalSteps} steps</span>
              </div>
              {isLocked ? (
                <span className="text-xs font-medium text-gray-500 bg-gray-100 px-2 py-1 rounded">PRO</span>
              ) : (
                <svg className="text-gray-400 group-hover:text-gray-600 transition-colors" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <line x1="5" y1="12" x2="19" y2="12"></line>
                  <polyline points="12 5 19 12 12 19"></polyline>
                </svg>
              )}
            </>
          )}
        </div>
      </div>
    </Link>
  )
}

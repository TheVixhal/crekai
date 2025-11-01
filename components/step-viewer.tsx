"use client"

import { useEffect, useState } from "react"
import { useRouter } from "next/navigation"
import Link from "next/link"
import ReactMarkdown from "react-markdown"
import { createClient } from "@/lib/supabase/client"
import { resetProjectProgress } from "@/app/projects/[slug]/action"

interface StepViewerProps {
  project: {
    id: string
    slug: string
    title: string
    total_steps: number
  }
  progress: any
  currentStep: number
  userId: string
}

export default function StepViewer({ project, progress, currentStep, userId }: StepViewerProps) {
  const router = useRouter()
  const [stepContent, setStepContent] = useState("")
  const [loading, setLoading] = useState(true)
  const [completed, setCompleted] = useState(false)
  const [error, setError] = useState("")
  const [resetting, setResetting] = useState(false)
  const supabase = createClient()

  useEffect(() => {
    loadStep()
  }, [currentStep])

  const loadStep = async () => {
    setLoading(true)
    setError("")

    try {
      // Fetch markdown file from public folder
      const response = await fetch(`/projects/${project.slug}/step_${currentStep}.md`)

      if (!response.ok) {
        setError(`Step ${currentStep} not found`)
        return
      }

      const content = await response.text()
      setStepContent(content)

      const completedSteps = progress?.completed_steps || []
      setCompleted(completedSteps.includes(currentStep))
    } catch (err) {
      setError("Failed to load step content")
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  const handleCompleteStep = async () => {
    try {
      const completedSteps = progress?.completed_steps || []

      if (!completedSteps.includes(currentStep)) {
        completedSteps.push(currentStep)
      }

      const nextStep = currentStep + 1
      const hasNextStep = nextStep <= project.total_steps

      if (!progress) {
        // Create new progress record
        const { error } = await supabase.from("user_progress").insert({
          user_id: userId,
          project_id: project.id,
          current_step: hasNextStep ? nextStep : currentStep,
          completed_steps: completedSteps,
        })

        if (error) throw error
      } else {
        // Update existing progress
        const { error } = await supabase
          .from("user_progress")
          .update({
            current_step: hasNextStep ? nextStep : currentStep,
            completed_steps: completedSteps,
            last_accessed: new Date().toISOString(),
          })
          .eq("user_id", userId)
          .eq("project_id", project.id)

        if (error) throw error
      }

      setCompleted(true)

      if (hasNextStep) {
        router.refresh()
      }
    } catch (err) {
      setError("Failed to save progress")
      console.error(err)
    }
  }

  const handleNextStep = () => {
    if (currentStep < project.total_steps) {
      router.push(`/projects/${project.slug}?step=${currentStep + 1}`)
    }
  }

  const handlePrevStep = () => {
    if (currentStep > 1) {
      router.push(`/projects/${project.slug}?step=${currentStep - 1}`)
    }
  }

  const handleResetProject = async () => {
    setResetting(true)
    try {
      await resetProjectProgress(project.id)
      router.push(`/projects/${project.slug}?step=1`)
      router.refresh()
    } catch (err) {
      setError("Failed to reset project")
      console.error(err)
    } finally {
      setResetting(false)
    }
  }

  const canGoNext = currentStep < project.total_steps && completed
  const canGoPrev = currentStep > 1
  const isProjectComplete = currentStep === project.total_steps && completed

  return (
    <div className="min-h-screen bg-amber-50">
      {/* Masthead */}
      <div className="border-b-4 border-black bg-white mb-8">
        <div className="max-w-4xl mx-auto px-6 py-6">
          <Link href="/projects" className="text-black hover:underline text-sm font-bold font-sans mb-4 inline-block">
            ← BACK TO PROJECTS
          </Link>
          <h1 className="text-4xl font-bold text-black font-serif mb-3">{project.title}</h1>
          <p className="text-gray-700 font-sans mb-4">
            Step {currentStep} of {project.total_steps}
          </p>
          <div className="w-full bg-gray-300 h-2 border-2 border-black">
            <div
              className="bg-black h-full transition-all"
              style={{
                width: `${(currentStep / project.total_steps) * 100}%`,
              }}
            />
          </div>
        </div>
      </div>

      {/* Main content */}
      <div className="max-w-4xl mx-auto px-6 pb-8">
        {isProjectComplete && (
          <div className="bg-green-50 border-3 border-green-700 p-8 mb-8 text-center">
            <h2 className="text-3xl font-bold text-green-900 font-serif mb-4">🎉 Project Complete!</h2>
            <p className="text-green-900 font-sans mb-6">
              Congratulations! You've completed all {project.total_steps} steps of {project.title}.
            </p>
            <button
              onClick={handleResetProject}
              disabled={resetting}
              className="px-8 py-3 bg-green-700 text-white font-serif font-bold border-2 border-green-900 hover:bg-green-800 disabled:opacity-50 transition"
            >
              {resetting ? "RESETTING..." : "START OVER"}
            </button>
          </div>
        )}

        {/* Content Card */}
        <div className="bg-white border-2 border-black p-8 mb-8">
          {loading ? (
            <div className="text-center text-gray-700 font-sans">Loading step...</div>
          ) : error ? (
            <div className="text-center text-red-900 font-sans font-bold">{error}</div>
          ) : (
            <div className="text-gray-900">
              <ReactMarkdown
                components={{
                  h1: ({ node, ...props }) => (
                    <h1 className="text-4xl font-bold text-black mb-4 font-serif" {...props} />
                  ),
                  h2: ({ node, ...props }) => (
                    <h2
                      className="text-3xl font-bold text-black mb-4 mt-8 font-serif border-b-2 border-black pb-2"
                      {...props}
                    />
                  ),
                  h3: ({ node, ...props }) => (
                    <h3 className="text-2xl font-bold text-black mb-3 mt-6 font-serif" {...props} />
                  ),
                  p: ({ node, children, ...props }) => {
                    return (
                      <p className="text-gray-900 mb-4 leading-relaxed font-serif" {...props}>
                        {children}
                      </p>
                    )
                  },
                  ul: ({ node, ...props }) => (
                    <ul className="list-disc list-inside text-gray-900 mb-4 space-y-2 font-sans" {...props} />
                  ),
                  ol: ({ node, ...props }) => (
                    <ol className="list-decimal list-inside text-gray-900 mb-4 space-y-2 font-sans" {...props} />
                  ),
                  code: ({ node, inline, className, children, ...props }: any) => {
                    if (inline) {
                      return (
                        <code
                          className="bg-gray-200 px-2 py-1 border border-black text-black font-mono text-sm"
                          {...props}
                        >
                          {children}
                        </code>
                      )
                    }

                    return (
                      <pre className="bg-gray-900 border-2 border-black p-4 text-gray-100 font-mono text-sm my-4 overflow-x-auto">
                        <code className="text-gray-100" {...props}>
                          {children}
                        </code>
                      </pre>
                    )
                  },
                  blockquote: ({ node, ...props }) => (
                    <blockquote
                      className="border-l-4 border-black pl-4 italic text-gray-700 my-4 font-serif"
                      {...props}
                    />
                  ),
                  a: ({ node, ...props }) => <a className="text-blue-900 hover:underline font-bold" {...props} />,
                }}
              >
                {stepContent}
              </ReactMarkdown>
            </div>
          )}
        </div>

        {/* Navigation */}
        <div className="flex justify-between items-center gap-4">
          <button
            onClick={handlePrevStep}
            disabled={!canGoPrev}
            className="px-6 py-2 bg-white border-2 border-black text-black font-serif font-bold disabled:opacity-30 hover:bg-gray-100 transition"
          >
            ← PREVIOUS
          </button>

          <button
            onClick={handleCompleteStep}
            disabled={completed}
            className={`px-8 py-2 font-serif font-bold border-2 border-black transition ${
              completed ? "bg-gray-300 text-gray-700 cursor-not-allowed" : "bg-green-600 text-white hover:bg-green-700"
            }`}
          >
            {completed ? "✓ COMPLETED" : "MARK COMPLETE"}
          </button>

          <button
            onClick={handleNextStep}
            disabled={!canGoNext}
            className={`px-6 py-2 font-serif font-bold border-2 border-black transition ${
              canGoNext ? "bg-black text-white hover:bg-gray-900" : "bg-gray-300 text-gray-700 cursor-not-allowed"
            }`}
          >
            NEXT →
          </button>
        </div>
      </div>
    </div>
  )
}

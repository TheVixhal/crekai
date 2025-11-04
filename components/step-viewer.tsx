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
      console.error(err)
      setError("Failed to load step content")
    } finally {
      setLoading(false)
    }
  }

  const handleCompleteStep = async () => {
    try {
      const completedSteps = progress?.completed_steps || []
      if (!completedSteps.includes(currentStep)) completedSteps.push(currentStep)
      const nextStep = currentStep + 1
      const hasNextStep = nextStep <= project.total_steps

      if (!progress) {
        const { error } = await supabase.from("user_progress").insert({
          user_id: userId,
          project_id: project.id,
          current_step: hasNextStep ? nextStep : currentStep,
          completed_steps: completedSteps,
        })
        if (error) throw error
      } else {
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
      if (hasNextStep) router.refresh()
    } catch (err) {
      console.error(err)
      setError("Failed to save progress")
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
      console.error(err)
      setError("Failed to reset project")
    } finally {
      setResetting(false)
    }
  }

  const canGoNext = currentStep < project.total_steps && completed
  const canGoPrev = currentStep > 1
  const isProjectComplete = currentStep === project.total_steps && completed

  return (
    <div className="min-h-screen bg-gradient-to-br from-[#f9fafb] via-[#f5f6f7] to-[#eef0f2] text-gray-900">
      {/* Header */}
      <header className="sticky top-0 z-40 backdrop-blur-md bg-white/80 border-b border-gray-200">
        <div className="max-w-4xl mx-auto px-6 py-5">
          <Link
            href="/projects"
            className="text-sm text-gray-600 hover:text-gray-900 transition font-medium"
          >
            ← Back to Projects
          </Link>
          <h1 className="text-3xl font-semibold text-gray-900 mt-2">{project.title}</h1>
          <p className="text-sm text-gray-500 mt-1">
            Step {currentStep} of {project.total_steps}
          </p>
          <div className="w-full bg-gray-200 rounded-full h-2 mt-3 overflow-hidden">
            <div
              className="bg-gradient-to-r from-indigo-500 to-blue-400 h-full transition-all duration-500"
              style={{ width: `${(currentStep / project.total_steps) * 100}%` }}
            />
          </div>
        </div>
      </header>

      <main className="max-w-4xl mx-auto px-6 py-10">
        {isProjectComplete && (
          <div className="bg-gradient-to-r from-green-50 to-emerald-100 border border-emerald-200 rounded-xl p-8 text-center shadow-sm mb-8">
            <h2 className="text-2xl font-semibold text-emerald-800 mb-3">🎉 Project Complete!</h2>
            <p className="text-emerald-700 mb-6 text-sm">
              You’ve completed all {project.total_steps} steps of <strong>{project.title}</strong>!
            </p>
            <button
              onClick={handleResetProject}
              disabled={resetting}
              className="px-6 py-2.5 rounded-md bg-emerald-700 text-white font-medium hover:bg-emerald-800 transition disabled:opacity-50"
            >
              {resetting ? "Resetting..." : "Start Over"}
            </button>
          </div>
        )}

        {/* Content Card */}
        <div className="bg-white/70 backdrop-blur-md border border-gray-200 rounded-2xl shadow-sm p-8 mb-10">
          {loading ? (
            <div className="text-center text-gray-500 text-sm">Loading step...</div>
          ) : error ? (
            <div className="text-center text-red-600 font-medium">{error}</div>
          ) : (
            <ReactMarkdown
              components={{
                h1: ({ ...props }) => (
                  <h1 className="text-3xl font-semibold text-gray-900 mb-4" {...props} />
                ),
                h2: ({ ...props }) => (
                  <h2 className="text-2xl font-semibold text-gray-900 mt-8 mb-3 border-b border-gray-200 pb-2" {...props} />
                ),
                h3: ({ ...props }) => (
                  <h3 className="text-xl font-semibold text-gray-800 mt-6 mb-2" {...props} />
                ),
                p: ({ children, ...props }) => (
                  <p className="text-gray-700 leading-relaxed mb-4 text-[15px]" {...props}>
                    {children}
                  </p>
                ),
                ul: ({ ...props }) => (
                  <ul className="list-disc list-inside text-gray-700 space-y-2 mb-4" {...props} />
                ),
                ol: ({ ...props }) => (
                  <ol className="list-decimal list-inside text-gray-700 space-y-2 mb-4" {...props} />
                ),
                code: ({ inline, className, children, ...props }: any) => {
                  if (inline) {
                    return (
                      <code className="bg-gray-100 px-2 py-1 rounded text-sm text-indigo-600 font-mono" {...props}>
                        {children}
                      </code>
                    )
                  }
                  return (
                    <pre className="bg-[#1e1e2f] text-gray-100 rounded-xl p-4 my-4 text-sm font-mono overflow-x-auto border border-gray-800 shadow-inner">
                      <code className="text-[13px]" {...props}>
                        {children}
                      </code>
                    </pre>
                  )
                },
                blockquote: ({ ...props }) => (
                  <blockquote className="border-l-4 border-indigo-400 pl-4 italic text-gray-700 my-4" {...props} />
                ),
                a: ({ ...props }) => (
                  <a className="text-indigo-600 hover:text-indigo-800 underline transition" {...props} />
                ),
              }}
            >
              {stepContent}
            </ReactMarkdown>
          )}
        </div>

        {/* Navigation */}
        <div className="flex justify-between items-center gap-4">
          <button
            onClick={handlePrevStep}
            disabled={!canGoPrev}
            className="px-5 py-2 rounded-md text-sm font-medium bg-white border border-gray-300 text-gray-800 hover:bg-gray-50 transition disabled:opacity-50"
          >
            ← Previous
          </button>

          <button
            onClick={handleCompleteStep}
            disabled={completed}
            className={`px-6 py-2 rounded-md text-sm font-medium transition ${
              completed
                ? "bg-gray-200 text-gray-600 cursor-not-allowed"
                : "bg-gradient-to-r from-green-600 to-emerald-500 text-white hover:from-green-700 hover:to-emerald-600"
            }`}
          >
            {completed ? "✓ Completed" : "Mark Complete"}
          </button>

          <button
            onClick={handleNextStep}
            disabled={!canGoNext}
            className={`px-5 py-2 rounded-md text-sm font-medium transition ${
              canGoNext
                ? "bg-gray-900 text-white hover:bg-gray-800"
                : "bg-gray-200 text-gray-600 cursor-not-allowed"
            }`}
          >
            Next →
          </button>
        </div>
      </main>
    </div>
  )
}

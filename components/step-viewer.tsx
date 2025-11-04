"use client"

import { useEffect, useState } from "react"
import { useRouter } from "next/navigation"
import Link from "next/link"
import ReactMarkdown from "react-markdown"
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism'
import { createClient } from "@/lib/supabase/client"
import { resetProjectProgress } from "@/app/projects/[slug]/action"
import Image from "next/image"
import ColabTokenDisplay from "./colab-token-display"

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
  const [stepConfig, setStepConfig] = useState<any>(null)
  const supabase = createClient()

  useEffect(() => {
    loadStep()
    loadProjectConfig()
  }, [currentStep])

  const loadProjectConfig = async () => {
    try {
      const response = await fetch(`/api/project-config/${project.slug}`)
      const data = await response.json()
      
      if (data.success && data.config) {
        const currentStepConfig = data.config.steps.find((s: any) => s.step === currentStep)
        setStepConfig(currentStepConfig || null)
      }
    } catch (err) {
      console.error("Failed to load project config:", err)
    }
  }

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

const handleManualComplete = async () => {
  // Only for steps without assignments
  try {
    const completedSteps = progress?.completed_steps || []

    if (!completedSteps.includes(currentStep)) {
      completedSteps.push(currentStep)
    }

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
    <div className="min-h-screen bg-[#F7F5F2] relative">
      {/* Grid Background */}
      <div 
        className="absolute inset-0 opacity-[0.15]" 
        style={{
          backgroundImage: `
            linear-gradient(to right, #D4D4D4 1px, transparent 1px),
            linear-gradient(to bottom, #D4D4D4 1px, transparent 1px)
          `,
          backgroundSize: '40px 40px'
        }}
      />

      {/* Header */}
      <div className="border-b border-gray-200 bg-white/80 backdrop-blur-sm sticky top-0 z-50 relative">
        <div className="max-w-5xl mx-auto px-6 py-2">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Link href="/projects" className="flex items-center gap-2 text-gray-600 hover:text-gray-900 transition text-sm font-medium">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <line x1="19" y1="12" x2="5" y2="12"></line>
                  <polyline points="12 19 5 12 12 5"></polyline>
                </svg>
                Back to Projects
              </Link>
              
              <div className="h-4 w-px bg-gray-300" />
              
              <h1 className="text-xl font-semibold text-gray-900">{project.title}</h1>
              
              <div className="h-4 w-px bg-gray-300" />
              
              <span className="text-sm text-gray-600">
                Step {currentStep} of {project.total_steps}
              </span>
            </div>
            
            <Image 
              src="/sphereo.png" 
              alt="CrekAI" 
              width={32} 
              height={32}
              className="rounded-lg"
            />
          </div>
        </div>
      </div>

      {/* Main content */}
      <div className="max-w-5xl mx-auto px-6 py-8 relative">
        {/* Colab Token Display - Only show if step has assignment */}
        {stepConfig?.has_assignment && (
          <ColabTokenDisplay projectSlug={project.slug} />
        )}

        {isProjectComplete && (
          <div className="bg-green-50 border border-green-200 rounded-xl p-8 mb-8 text-center">
            <div className="text-6xl mb-4">🎉</div>
            <h2 className="text-3xl font-semibold text-green-900 mb-3">Project Complete!</h2>
            <p className="text-green-800 mb-6">
              Congratulations! You've completed all {project.total_steps} steps of {project.title}.
            </p>
            <button
              onClick={handleResetProject}
              disabled={resetting}
              className="px-6 py-3 bg-green-600 text-white font-medium rounded-lg hover:bg-green-700 disabled:opacity-50 transition"
            >
              {resetting ? "Resetting..." : "Start Over"}
            </button>
          </div>
        )}

        {/* Content Card */}
        <div className="bg-white border border-gray-200 rounded-xl p-8 mb-8 shadow-sm">
          {loading ? (
            <div className="text-center text-gray-600 py-12">Loading step...</div>
          ) : error ? (
            <div className="text-center text-red-600 py-12 font-medium">{error}</div>
          ) : (
            <div className="prose prose-gray max-w-none">
              <ReactMarkdown
                components={{
                  h1: ({ node, ...props }) => (
                    <h1 className="text-3xl font-semibold text-gray-900 mb-6 mt-0" {...props} />
                  ),
                  h2: ({ node, ...props }) => (
                    <h2 className="text-2xl font-semibold text-gray-900 mb-4 mt-8 pb-2 border-b border-gray-200" {...props} />
                  ),
                  h3: ({ node, ...props }) => (
                    <h3 className="text-xl font-semibold text-gray-900 mb-3 mt-6" {...props} />
                  ),
                  p: ({ node, ...props }) => (
                    <p className="text-gray-700 mb-4 leading-relaxed" {...props} />
                  ),
                  ul: ({ node, ...props }) => (
                    <ul className="list-disc list-outside ml-6 text-gray-700 mb-4 space-y-2" {...props} />
                  ),
                  ol: ({ node, ...props }) => (
                    <ol className="list-decimal list-outside ml-6 text-gray-700 mb-4 space-y-2" {...props} />
                  ),
                  li: ({ node, ...props }) => (
                    <li className="text-gray-700" {...props} />
                  ),
                  code: ({ node, inline, className, children, ...props }: any) => {
                    const match = /language-(\w+)/.exec(className || '')
                    const language = match ? match[1] : ''
                    
                    if (inline) {
                      return (
                        <code className="bg-gray-100 px-2 py-0.5 rounded text-sm font-mono text-gray-900 border border-gray-200" {...props}>
                          {children}
                        </code>
                      )
                    }

                    return (
                      <div className="my-6 rounded-lg overflow-hidden border border-gray-200">
                        {language && (
                          <div className="bg-gray-100 px-4 py-2 text-xs font-medium text-gray-600 border-b border-gray-200">
                            {language}
                          </div>
                        )}
                        <SyntaxHighlighter
                          language={language || 'text'}
                          style={vscDarkPlus}
                          customStyle={{
                            margin: 0,
                            borderRadius: 0,
                            fontSize: '0.875rem',
                            padding: '1.25rem',
                          }}
                          showLineNumbers={true}
                        >
                          {String(children).replace(/\n$/, '')}
                        </SyntaxHighlighter>
                      </div>
                    )
                  },
                  pre: ({ node, ...props }) => <div {...props} />,
                  blockquote: ({ node, ...props }) => (
                    <blockquote className="border-l-4 border-gray-300 pl-4 italic text-gray-600 my-6" {...props} />
                  ),
                  a: ({ node, ...props }) => (
                    <a className="text-blue-600 hover:text-blue-800 underline" {...props} />
                  ),
                  strong: ({ node, ...props }) => (
                    <strong className="font-semibold text-gray-900" {...props} />
                  ),
                  em: ({ node, ...props }) => (
                    <em className="italic" {...props} />
                  ),
                }}
              >
                {stepContent}
              </ReactMarkdown>
            </div>
          )}
        </div>

        {/* Navigation */}
        <div className="space-y-4">
          {/* Colab Verification Status - Only for assignment steps */}
          {!completed && stepConfig?.has_assignment && (
            <div className="bg-amber-50 border border-amber-200 rounded-lg p-4 text-center">
              <div className="flex items-center justify-center gap-2 text-amber-800 mb-2">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path>
                  <line x1="12" y1="9" x2="12" y2="13"></line>
                  <line x1="12" y1="17" x2="12.01" y2="17"></line>
                </svg>
                <span className="font-semibold">Complete Assignment in Colab to Unlock</span>
              </div>
              <p className="text-sm text-amber-700">
                Run the verification cell in your Colab notebook to mark this step as complete
              </p>
            </div>
          )}
          
          <div className="flex justify-between items-center gap-4">
            <button
              onClick={handlePrevStep}
              disabled={!canGoPrev}
              className="px-6 py-3 bg-white border border-gray-200 text-gray-700 font-medium rounded-lg disabled:opacity-40 disabled:cursor-not-allowed hover:bg-gray-50 hover:border-gray-300 transition flex items-center gap-2"
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <line x1="19" y1="12" x2="5" y2="12"></line>
                <polyline points="12 19 5 12 12 5"></polyline>
              </svg>
              Previous
            </button>

            {/* Show different middle button based on step type */}
            {stepConfig?.has_assignment ? (
              <div className={`px-8 py-3 font-medium rounded-lg flex items-center gap-2 ${
                completed 
                  ? "bg-green-100 text-green-700 border border-green-300" 
                  : "bg-gray-100 text-gray-500 border border-gray-200"
              }`}>
                {completed ? (
                  <>
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path>
                      <polyline points="22 4 12 14.01 9 11.01"></polyline>
                    </svg>
                    Verified via Colab
                  </>
                ) : (
                  <>
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <circle cx="12" cy="12" r="10"></circle>
                      <line x1="12" y1="8" x2="12" y2="12"></line>
                      <line x1="12" y1="16" x2="12.01" y2="16"></line>
                    </svg>
                    Awaiting Verification
                  </>
                )}
              </div>
            ) : (
              <button
                onClick={handleManualComplete}
                disabled={completed}
                className={`px-8 py-3 font-medium rounded-lg transition ${
                  completed 
                    ? "bg-gray-100 text-gray-500 cursor-not-allowed border border-gray-200" 
                    : "bg-green-600 text-white hover:bg-green-700"
                }`}
              >
                {completed ? "✓ Completed" : "Mark Complete"}
              </button>
            )}

            <button
              onClick={handleNextStep}
              disabled={!canGoNext}
              className={`px-6 py-3 font-medium rounded-lg transition flex items-center gap-2 ${
                canGoNext 
                  ? "bg-gray-900 text-white hover:bg-gray-800" 
                  : "bg-gray-100 text-gray-400 cursor-not-allowed border border-gray-200"
              }`}
            >
              Next
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <line x1="5" y1="12" x2="19" y2="12"></line>
                <polyline points="12 5 19 12 12 19"></polyline>
              </svg>
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

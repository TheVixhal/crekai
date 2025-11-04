import { createClient } from "@/lib/supabase/server"
import { redirect } from "next/navigation"
import { Lock } from "lucide-react"

export default async function ProjectsPage() {
  const supabase = await createClient()
  const {
    data: { user },
  } = await supabase.auth.getUser()

  if (!user) {
    redirect("/auth/login")
  }

  // Fetch all projects
  const { data: projects, error: projectsError } = await supabase
    .from("projects")
    .select("*")
    .order("created_at", { ascending: true })

  if (projectsError) {
    console.error("Error fetching projects:", projectsError)
  }

  // Fetch user progress
  const { data: progressData } = await supabase
    .from("user_progress")
    .select("*")
    .eq("user_id", user.id)

  const progressMap = (progressData || []).reduce((acc: Record<string, any>, prog: any) => {
    acc[prog.project_id] = prog
    return acc
  }, {})

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200">
        <div className="max-w-6xl mx-auto flex items-center justify-between px-6 py-4">
          <div className="flex items-center gap-8">
            <div className="flex items-center gap-2">
              <div className="w-10 h-10 bg-gray-900 rounded-lg flex items-center justify-center">
                <span className="text-white text-xl font-bold">{"</>"}</span>
              </div>
              <h1 className="text-xl font-semibold text-gray-900">CodeCrafters</h1>
            </div>
            <nav className="hidden md:flex items-center gap-6">
              <a href="#" className="text-sm font-medium text-teal-500 border-b-2 border-teal-500 pb-4">
                Catalog
              </a>
              <a href="#" className="text-sm font-medium text-gray-600 hover:text-gray-900 pb-4">
                Roadmap
              </a>
              <a href="#" className="text-sm font-medium text-gray-600 hover:text-gray-900 pb-4">
                Leaderboard
              </a>
            </nav>
          </div>
          <div className="flex items-center gap-4">
            <button className="text-sm text-gray-600 hover:text-gray-900">
              Feedback
            </button>
            <button className="px-4 py-2 bg-teal-500 text-white rounded-md text-sm font-medium hover:bg-teal-600 transition">
              Upgrade ⭐
            </button>
            <form
              action={async () => {
                "use server"
                const supabase = await createClient()
                await supabase.auth.signOut()
                redirect("/")
              }}
            >
              <button className="text-sm text-gray-600 hover:text-gray-900">
                Sign Out
              </button>
            </form>
          </div>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-6 py-12">
        {/* Page Title */}
        <h2 className="text-4xl font-bold text-gray-800 mb-12">Challenges</h2>

        {/* Projects Grid */}
        <div className="grid md:grid-cols-2 gap-6">
          {projects && projects.length > 0 ? (
            projects.map((project: any) => {
              const progress = progressMap[project.id]
              const totalSteps = project.total_steps || 0
              const completedSteps = progress?.completed_steps?.length || 0
              const isLocked = !progress && project.is_premium
              const isFree = project.is_free_this_month

              return (
                <a
                  key={project.id}
                  href={isLocked ? "#" : `/projects/${project.id}`}
                  className={`group relative bg-white rounded-lg border border-gray-200 p-6 transition-all hover:shadow-lg ${
                    isLocked ? "cursor-not-allowed opacity-75" : "hover:border-gray-300"
                  }`}
                >
                  <div className="flex justify-between items-start">
                    <div className="flex-1">
                      <h3 className="text-xl font-semibold text-gray-800 mb-3 group-hover:text-gray-900">
                        {project.title}
                      </h3>
                      <p className="text-sm text-gray-600 leading-relaxed mb-6">
                        {project.description}
                      </p>

                      {/* Status Section */}
                      <div className="flex items-center gap-3">
                        {isFree && !isLocked && (
                          <span className="text-xs font-bold text-teal-500 uppercase tracking-wide">
                            FREE THIS MONTH
                          </span>
                        )}
                        {progress && totalSteps > 0 && (
                          <div className="flex items-center gap-2 text-sm text-gray-500">
                            <div className="flex items-center gap-1">
                              <svg
                                className="w-4 h-4 text-teal-400"
                                fill="none"
                                stroke="currentColor"
                                viewBox="0 0 24 24"
                              >
                                <path
                                  strokeLinecap="round"
                                  strokeLinejoin="round"
                                  strokeWidth={2}
                                  d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                                />
                              </svg>
                              <span className="font-medium">
                                {completedSteps}/{totalSteps} stages
                              </span>
                            </div>
                          </div>
                        )}
                        {!progress && !isLocked && totalSteps > 0 && (
                          <div className="flex items-center gap-2 text-sm text-gray-400">
                            <svg
                              className="w-4 h-4"
                              fill="none"
                              stroke="currentColor"
                              viewBox="0 0 24 24"
                            >
                              <path
                                strokeLinecap="round"
                                strokeLinejoin="round"
                                strokeWidth={2}
                                d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"
                              />
                            </svg>
                            <span>{totalSteps} stages</span>
                          </div>
                        )}
                        {isLocked && (
                          <div className="flex items-center gap-2 text-sm text-gray-400">
                            <Lock className="w-4 h-4" />
                          </div>
                        )}
                      </div>
                    </div>

                    {/* Icon */}
                    <div className="ml-4">
                      <div className="w-12 h-12 rounded-lg bg-gradient-to-br from-purple-100 to-purple-200 flex items-center justify-center text-2xl">
                        {project.icon || "📦"}
                      </div>
                    </div>
                  </div>

                  {isLocked && (
                    <div className="absolute top-4 right-4">
                      <Lock className="w-5 h-5 text-gray-300" />
                    </div>
                  )}
                </a>
              )
            })
          ) : (
            <div className="col-span-full bg-white border border-gray-200 rounded-lg p-12 text-center">
              <div className="text-6xl mb-4">📚</div>
              <p className="text-gray-800 font-medium text-lg">
                No projects available yet.
              </p>
              <p className="text-gray-500 text-sm mt-2">
                Check back soon for new learning opportunities.
              </p>
            </div>
          )}
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-gray-200 mt-20 py-8 bg-white">
        <div className="max-w-6xl mx-auto px-6 flex flex-col md:flex-row justify-between items-center gap-4">
          <p className="text-gray-500 text-sm">
            © 2025 CrekAI — Empowering learners through hands-on AI experiences.
          </p>
          <div className="flex gap-6 text-sm">
            <a href="#" className="text-gray-600 hover:text-gray-900 transition">
              About
            </a>
            <a href="#" className="text-gray-600 hover:text-gray-900 transition">
              Help
            </a>
            <a href="#" className="text-gray-600 hover:text-gray-900 transition">
              Contact
            </a>
          </div>
        </div>
      </footer>
    </div>
  )
}

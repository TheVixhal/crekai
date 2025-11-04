import { createClient } from "@/lib/supabase/server"
import { redirect } from "next/navigation"
import Link from "next/link"

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

  // Fetch user progress for all projects
  const { data: progressData, error: progressError } = await supabase
    .from("user_progress")
    .select("*")
    .eq("user_id", user.id)

  if (progressError) {
    console.error("Error fetching user progress:", progressError)
  }

  // Map progress data
  const progressMap = (progressData || []).reduce((acc: Record<string, any>, prog: any) => {
    acc[prog.project_id] = prog
    return acc
  }, {})

  return (
    <div className="min-h-screen bg-[#f9fafb] flex flex-col text-gray-900">
      {/* Navbar */}
      <header className="border-b border-gray-200 bg-white">
        <div className="max-w-7xl mx-auto px-6 py-3 flex items-center justify-between">
          <div className="flex items-center gap-10">
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-full bg-gradient-to-r from-indigo-500 to-purple-500 flex items-center justify-center text-white font-bold">
                AI
              </div>
              <span className="font-semibold text-gray-800 text-lg">CrekAI</span>
            </div>
            <nav className="hidden md:flex items-center gap-6 text-sm font-medium text-gray-600">
              <a href="#" className="hover:text-gray-900 transition">Catalog</a>
              <a href="#" className="hover:text-gray-900 transition">Roadmap</a>
              <a href="#" className="hover:text-gray-900 transition">Leaderboard</a>
            </nav>
          </div>

          <div className="flex items-center gap-4">
            <button className="hidden md:block px-4 py-1.5 bg-emerald-500 text-white text-sm rounded-md font-medium hover:bg-emerald-600 transition">
              Upgrade
            </button>
            <form
              action={async () => {
                "use server"
                const supabase = await createClient()
                await supabase.auth.signOut()
                redirect("/")
              }}
            >
              <button className="text-sm font-medium text-gray-600 hover:text-gray-900 transition">
                {user?.email?.split("@")[0] || "Account"} ▼
              </button>
            </form>
          </div>
        </div>
      </header>

      {/* Main Section */}
      <main className="max-w-7xl mx-auto w-full px-6 py-10 flex-grow">
        <h1 className="text-2xl font-semibold text-gray-900 mb-8">Projects</h1>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {projects && projects.length > 0 ? (
            projects.map((project: any) => {
              const progress = progressMap[project.id] || {}
              const completed = progress.completed_steps?.length || 0
              const totalSteps = project.total_steps || 0
              const isCompleted = totalSteps > 0 && completed >= totalSteps

              return (
                <div
                  key={project.id}
                  className="bg-white border border-gray-200 rounded-lg p-5 shadow-sm hover:shadow-md transition-all"
                >
                  <div className="flex justify-between items-start">
                    <div>
                      <h2 className="font-semibold text-gray-900 text-lg mb-1">
                        {project.title}
                      </h2>
                      <p className="text-gray-600 text-sm leading-relaxed line-clamp-2">
                        {project.description}
                      </p>
                    </div>
                    <div className="text-gray-400 text-xl">
                      {project.icon || "📘"}
                    </div>
                  </div>

                  <div className="mt-4 flex items-center justify-between text-sm text-gray-500">
                    <span className="flex items-center gap-1">
                      <svg
                        xmlns="http://www.w3.org/2000/svg"
                        fill="none"
                        viewBox="0 0 24 24"
                        strokeWidth={2}
                        stroke="currentColor"
                        className="w-4 h-4 text-indigo-500"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          d="M12 6v6h4.5m4.5 0a9 9 0 11-18 0 9 9 0 0118 0z"
                        />
                      </svg>
                      {totalSteps > 0 ? `${completed}/${totalSteps} stages` : "0 stages"}
                    </span>

                    <Link
                      href={`/projects/${project.id}`}
                      className={`font-medium transition ${
                        isCompleted
                          ? "text-green-600 hover:text-green-700"
                          : "text-indigo-600 hover:text-indigo-700"
                      }`}
                    >
                      {isCompleted ? "Completed" : "Continue →"}
                    </Link>
                  </div>
                </div>
              )
            })
          ) : (
            <div className="col-span-full bg-white border border-gray-200 rounded-lg p-10 text-center shadow-sm">
              <div className="text-5xl mb-3">📚</div>
              <p className="text-gray-800 font-medium">No projects available yet.</p>
              <p className="text-gray-500 text-sm mt-1">
                Check back soon for new learning opportunities.
              </p>
            </div>
          )}
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-gray-200 bg-white py-6">
        <div className="max-w-7xl mx-auto px-6 flex flex-col md:flex-row justify-between items-center gap-4">
          <p className="text-gray-500 text-sm">© 2025 CrekAI — Learn by building.</p>
          <div className="flex gap-5 text-sm">
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

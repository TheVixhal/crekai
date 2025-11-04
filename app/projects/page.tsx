import { createClient } from "@/lib/supabase/server"
import { redirect } from "next/navigation"
import ProjectCard from "@/components/project-card"

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
    <div className="min-h-screen bg-gray-50 text-gray-900 flex flex-col">
      {/* Header */}
      <header className="sticky top-0 z-50 backdrop-blur-md bg-white/80 border-b border-gray-200">
        <div className="max-w-6xl mx-auto flex items-center justify-between px-6 py-4">
          <h1 className="text-xl font-semibold tracking-tight text-gray-900">CrekAI</h1>
          <form
            action={async () => {
              "use server"
              const supabase = await createClient()
              await supabase.auth.signOut()
              redirect("/")
            }}
          >
            <button className="px-4 py-2 bg-gray-900 text-white rounded-md text-sm font-medium hover:bg-gray-800 transition">
              Sign Out
            </button>
          </form>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-grow max-w-6xl mx-auto w-full px-6 py-10">
        <div className="mb-8">
          <h2 className="text-2xl font-semibold text-gray-900">Projects</h2>
          <p className="text-gray-600 text-sm mt-1">
            Choose a project and start exploring.
          </p>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {projects && projects.length > 0 ? (
            projects.map((project: any) => (
              <ProjectCard
                key={project.id}
                project={project}
                progress={progressMap[project.id]}
              />
            ))
          ) : (
            <div className="col-span-full bg-white border border-gray-200 rounded-xl p-10 text-center shadow-sm">
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
      <footer className="border-t border-gray-200 bg-white/60 backdrop-blur-sm py-6">
        <div className="max-w-6xl mx-auto px-6 flex flex-col md:flex-row justify-between items-center gap-4">
          <p className="text-gray-500 text-sm">
            © 2025 CrekAI — Learn by building.
          </p>
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

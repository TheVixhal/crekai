import { createClient } from "@/lib/supabase/server"
import { redirect } from "next/navigation"
import ProjectCard from "@/components/project-card"
import ProfileSidebar from "@/components/profile-sidebar"

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

  const { data: userProfile } = await supabase
    .from("user_profiles")
    .select("*")
    .eq("id", user.id)
    .maybeSingle()

  const projectsEnrolled = Object.keys(progressMap).length
  const completedSteps = Object.values(progressMap).reduce((sum, prog: any) => {
    return sum + (prog.completed_steps?.length || 0)
  }, 0)
  const projectsInProgress = Object.values(progressMap).filter((prog: any) => {
    return prog.current_step && prog.current_step <= (prog.total_steps || 0)
  }).length

  return (
    <div className="min-h-screen bg-gradient-to-br from-white via-gray-50 to-gray-100 text-gray-900">
      {/* Header */}
      <header className="sticky top-0 z-50 backdrop-blur-md bg-white/80 border-b border-gray-200">
        <div className="max-w-7xl mx-auto flex items-center justify-between px-6 py-4">
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-semibold tracking-tight text-gray-900">CrekAI</h1>
            <span className="text-sm text-gray-500">Learning Projects</span>
          </div>
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

      <main className="max-w-7xl mx-auto px-6 py-10 grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Main content */}
        <section className="lg:col-span-2 space-y-8">
          {/* Welcome banner */}
          <div className="rounded-2xl bg-gradient-to-r from-blue-50 to-indigo-100 p-8 border border-gray-200 shadow-sm">
            <div className="flex justify-between items-center">
              <div>
                <h2 className="text-2xl font-semibold text-gray-900 mb-2">
                  Welcome back, {userProfile?.full_name?.split(" ")[0] || "Learner"} 👋
                </h2>
                <p className="text-gray-600 text-sm">
                  Continue your journey in building real-world AI projects.
                </p>
              </div>
              <div className="hidden md:block text-5xl opacity-80">🚀</div>
            </div>
          </div>

          {/* Section Header */}
          <div className="space-y-2">
            <h3 className="text-xl font-semibold text-gray-900">Featured Projects</h3>
            <p className="text-gray-600 text-sm">
              Choose a project and start building something impactful.
            </p>
          </div>

          {/* Projects Grid */}
          <div className="grid md:grid-cols-2 gap-6">
            {projects && projects.length > 0 ? (
              projects.map((project: any) => (
                <ProjectCard key={project.id} project={project} progress={progressMap[project.id]} />
              ))
            ) : (
              <div className="col-span-full bg-white border border-gray-200 rounded-xl p-10 text-center shadow-sm">
                <div className="text-5xl mb-3">📚</div>
                <p className="text-gray-800 font-medium">
                  No projects available yet.
                </p>
                <p className="text-gray-500 text-sm mt-1">
                  Check back soon for new learning opportunities.
                </p>
              </div>
            )}
          </div>
        </section>

        {/* Sidebar */}
        <aside className="lg:col-span-1">
          <div className="sticky top-28">
            <ProfileSidebar
              user={user}
              userProfile={userProfile}
              createdAt={userProfile?.created_at}
              projectsEnrolled={projectsEnrolled}
              completedSteps={completedSteps}
              projectsInProgress={projectsInProgress}
            />
          </div>
        </aside>
      </main>

      {/* Footer */}
      <footer className="border-t border-gray-200 mt-16 py-8 bg-white/60 backdrop-blur-sm">
        <div className="max-w-7xl mx-auto px-6 flex flex-col md:flex-row justify-between items-center gap-4">
          <p className="text-gray-500 text-sm">
            © 2025 CrekAI — Empowering learners through hands-on AI experiences.
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

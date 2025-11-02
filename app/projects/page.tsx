import { createClient } from "@/lib/supabase/server"
import { redirect } from "next/navigation"
import { unstable_cache } from "next/cache"
import ProjectCard from "@/components/project-card"
import ProfileSidebar from "@/components/profile-sidebar"

// Cache projects for 5 minutes (revalidate every 300 seconds)
const getCachedProjects = unstable_cache(
  async () => {
    const supabase = await createClient()
    const { data: projects, error } = await supabase
      .from("projects")
      .select("*")
      .order("created_at", { ascending: false }) // Show newest first
      .limit(50) // Limit to 50 projects for performance
    
    if (error) throw error
    return projects
  },
  ["projects-list"],
  { revalidate: 300, tags: ["projects"] }
)

export default async function ProjectsPage() {
  const supabase = await createClient()
  const {
    data: { user },
  } = await supabase.auth.getUser()

  if (!user) {
    redirect("/auth/login")
  }

  // Parallel data fetching instead of sequential
  const [projects, progressData, userProfile] = await Promise.all([
    getCachedProjects().catch(() => []),
    supabase
      .from("user_progress")
      .select("project_id, current_step, total_steps, completed_steps")
      .eq("user_id", user.id)
      .then(({ data }) => data || []),
    supabase
      .from("user_profiles")
      .select("id, created_at, display_name, avatar_url")
      .eq("id", user.id)
      .maybeSingle()
      .then(({ data }) => data)
  ])

  // Optimize progress map creation
  const progressMap = progressData.reduce((acc, prog) => {
    acc[prog.project_id] = prog
    return acc
  }, {} as Record<string, any>)

  // Calculate stats efficiently
  const projectsEnrolled = progressData.length
  const completedSteps = progressData.reduce((sum, prog) => 
    sum + (prog.completed_steps?.length || 0), 0
  )
  const projectsInProgress = progressData.filter((prog) => 
    prog.current_step && prog.current_step <= (prog.total_steps || 0)
  ).length

  return (
    <div className="min-h-screen bg-gradient-to-br from-amber-50 via-orange-50 to-rose-50">
      {/* Enhanced Masthead with gradient and shadow */}
      <div className="border-b-4 border-black bg-gradient-to-r from-white to-amber-50 sticky top-0 z-50 shadow-lg backdrop-blur-sm bg-opacity-95">
        <div className="max-w-7xl mx-auto px-6 py-4 flex justify-between items-center">
          <div className="flex items-center gap-4">
            <div className="w-12 h-12 bg-gradient-to-br from-orange-400 to-rose-500 rounded-lg border-2 border-black flex items-center justify-center">
              <span className="text-white font-bold text-xl">C</span>
            </div>
            <div>
              <h1 className="text-4xl font-bold text-black font-serif tracking-tight">CrekAI</h1>
              <p className="text-xs text-gray-600 uppercase tracking-widest font-medium">Learning Projects</p>
            </div>
          </div>
          <form
            action={async () => {
              "use server"
              const supabase = await createClient()
              await supabase.auth.signOut()
              redirect("/")
            }}
          >
            <button className="px-6 py-2.5 border-2 border-black bg-white text-black font-sans font-bold hover:bg-black hover:text-white transition-all duration-300 shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] hover:shadow-[2px_2px_0px_0px_rgba(0,0,0,1)] hover:translate-x-[2px] hover:translate-y-[2px] active:shadow-none active:translate-x-[4px] active:translate-y-[4px]">
              Sign Out
            </button>
          </form>
        </div>
      </div>

      {/* Content with improved spacing */}
      <div className="max-w-7xl mx-auto p-6 lg:p-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 lg:gap-10">
          {/* Main Content with enhanced header */}
          <div className="lg:col-span-2 space-y-8">
            {/* Welcome Banner */}
            <div className="bg-gradient-to-r from-orange-400 to-rose-500 border-4 border-black p-8 shadow-[8px_8px_0px_0px_rgba(0,0,0,1)]">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-3xl font-bold text-white font-serif mb-2 drop-shadow-lg">
                    Welcome Back, {user.email?.split('@')[0]}!
                  </h2>
                  <p className="text-white/90 font-sans text-lg">
                    Continue your AI/ML learning journey
                  </p>
                </div>
                <div className="hidden md:block text-6xl">🚀</div>
              </div>
            </div>

            {/* Section Header */}
            <div className="space-y-3">
              <div className="flex items-center gap-3">
                <div className="h-1 w-12 bg-gradient-to-r from-orange-400 to-rose-500"></div>
                <h2 className="text-3xl font-bold text-black font-serif">Featured Projects</h2>
              </div>
              <p className="text-gray-700 font-sans text-lg pl-15">
                Choose a project and start building something amazing
              </p>
            </div>

            {/* Projects Grid with improved spacing */}
            <div className="grid md:grid-cols-2 gap-6">
              {projects && projects.length > 0 ? (
                projects.map((project: any) => (
                  <ProjectCard key={project.id} project={project} progress={progressMap[project.id]} />
                ))
              ) : (
                <div className="col-span-full bg-white border-4 border-black p-12 text-center shadow-[8px_8px_0px_0px_rgba(0,0,0,1)]">
                  <div className="text-6xl mb-4">📚</div>
                  <p className="text-gray-700 font-sans text-lg font-medium">
                    No projects available yet.
                  </p>
                  <p className="text-gray-500 font-sans text-sm mt-2">
                    Check back soon for exciting learning opportunities!
                  </p>
                </div>
              )}
            </div>

            {/* Pagination hint - can be implemented later */}
            {projects && projects.length === 50 && (
              <div className="text-center py-4">
                <p className="text-gray-600 text-sm">Showing 50 most recent projects</p>
              </div>
            )}
          </div>

          {/* Profile Sidebar with sticky positioning */}
          <div className="lg:col-span-1">
            <div className="sticky top-24">
              <ProfileSidebar
                user={user}
                userProfile={userProfile}
                createdAt={userProfile?.created_at}
                projectsEnrolled={projectsEnrolled}
                completedSteps={completedSteps}
                projectsInProgress={projectsInProgress}
              />
            </div>
          </div>
        </div>
      </div>

      {/* Footer */}
      <div className="mt-16 border-t-4 border-black bg-white">
        <div className="max-w-7xl mx-auto px-6 py-8">
          <div className="flex flex-col md:flex-row justify-between items-center gap-4">
            <p className="text-gray-600 font-sans text-sm">
              © 2024 CrekAI. Empowering learners through hands-on AI projects.
            </p>
            <div className="flex gap-4">
              <a href="#" className="text-gray-600 hover:text-black transition font-sans text-sm font-medium">
                About
              </a>
              <a href="#" className="text-gray-600 hover:text-black transition font-sans text-sm font-medium">
                Help
              </a>
              <a href="#" className="text-gray-600 hover:text-black transition font-sans text-sm font-medium">
                Contact
              </a>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

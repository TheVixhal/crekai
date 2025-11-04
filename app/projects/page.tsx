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

  // Fetch user progress for all projects
  const { data: progressData } = await supabase.from("user_progress").select("*").eq("user_id", user.id)

  const progressMap = (progressData || []).reduce((acc: Record<string, any>, prog: any) => {
    acc[prog.project_id] = prog
    return acc
  }, {})

  const { data: userProfile } = await supabase.from("user_profiles").select("*").eq("id", user.id).maybeSingle()

  const projectsEnrolled = Object.keys(progressMap).length
  const completedSteps = Object.values(progressMap).reduce((sum, prog: any) => {
    return sum + (prog.completed_steps?.length || 0)
  }, 0)
  const projectsInProgress = Object.values(progressMap).filter((prog: any) => {
    return prog.current_step && prog.current_step <= (prog.total_steps || 0)
  }).length

  return (
    <div className="min-h-screen bg-gradient-to-br from-amber-50 via-orange-50 to-rose-50">
      {/* Enhanced Masthead with gradient and shadow */}
      <div className="border-b-4 border-black bg-gradient-to-r from-white to-amber-50 sticky top-0 z-50 shadow-lg backdrop-blur-sm bg-opacity-95">
        <div className="max-w-7xl mx-auto px-6 py-4 flex justify-between items-center">
          <div className="flex items-center gap-4">
           
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
            <div className="bg-gradient-to-r from-cyan-400 to-cyan-600 border-4 border-black p-8 shadow-[8px_8px_0px_0px_rgba(0,0,0,1)]">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-3xl font-bold text-white font-serif mb-2 drop-shadow-lg">
                    Welcome Back, {userProfile.full_name}!
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
                
                <h2 className="text-3xl font-bold text-black font-serif">Featured Projects</h2>
              </div>
              <p className="text-gray-700 font-sans text-lg">
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
              © 2025 CrekAI. Empowering learners through hands-on AI projects.
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

//// i dont want this page retro style make this anthropic style

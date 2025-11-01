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
    <div className="min-h-screen bg-amber-50">
      {/* Masthead */}
      <div className="border-b-4 border-black bg-white sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 py-4 flex justify-between items-start">
          <div>
            <h1 className="text-5xl font-bold text-black font-serif">CrekAI</h1>
            <p className="text-xs text-gray-700 uppercase tracking-widest">Learning Projects</p>
          </div>
          <form
            action={async () => {
              "use server"
              const supabase = await createClient()
              await supabase.auth.signOut()
              redirect("/")
            }}
          >
            <button className="px-6 py-2 border-2 border-black bg-white text-black font-sans font-bold hover:bg-gray-100 transition">
              Sign Out
            </button>
          </form>
        </div>
      </div>

      {/* Content */}
      <div className="max-w-7xl mx-auto p-6">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Main Content */}
          <div className="lg:col-span-2">
            <div className="mb-8">
              <h2 className="text-2xl font-bold text-black font-serif mb-2">Featured Projects</h2>
              <p className="text-gray-700 font-sans">Choose a project and begin your AI/ML journey</p>
            </div>

            {/* Projects Grid */}
            <div className="grid md:grid-cols-2 gap-6">
              {projects && projects.length > 0 ? (
                projects.map((project: any) => (
                  <ProjectCard key={project.id} project={project} progress={progressMap[project.id]} />
                ))
              ) : (
                <div className="col-span-full bg-white border-2 border-black p-8 text-center">
                  <p className="text-gray-700 font-sans">No projects available yet.</p>
                </div>
              )}
            </div>
          </div>

          {/* Profile Sidebar */}
          <div className="lg:col-span-1">
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
  )
}

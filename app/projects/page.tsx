import { createClient } from "@/lib/supabase/server"
import { redirect } from "next/navigation"
import ProjectCard from "@/components/project-card"
import ProfileSidebar from "@/components/profile-sidebar"
import Image from "next/image"

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

  // Separate projects and learning modules
  const learningModules = projects?.filter((p: any) => p.isLearningModule === true) || []
  const regularProjects = projects?.filter((p: any) => p.isLearningModule !== true) || []

  // Fetch user progress for all projects
  const { data: progressData } = await supabase.from("user_progress").select("*").eq("user_id", user.id)

  const progressMap = (progressData || []).reduce((acc: Record<string, any>, prog: any) => {
    acc[prog.project_id] = prog
    return acc
  }, {})

  const { data: userProfile } = await supabase.from("user_profiles").select("*").eq("id", user.id).maybeSingle()

  const userLevel = userProfile?.level ?? 0

  const projectsEnrolled = Object.keys(progressMap).length
  const completedSteps = Object.values(progressMap).reduce((sum, prog: any) => {
    return sum + (prog.completed_steps?.length || 0)
  }, 0)
  const projectsInProgress = Object.values(progressMap).filter((prog: any) => {
    return prog.current_step && prog.current_step <= (prog.total_steps || 0)
  }).length

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
        <div className="max-w-7xl mx-auto px-6 py-4 flex justify-between items-center">
          <div className="flex items-center gap-3">
            <Image 
              src="/sphereo.png" 
              alt="CrekAI Logo" 
              width={40} 
              height={40}
              className="rounded-lg"
            />
            <h1 className="text-2xl font-semibold text-gray-900">CrekAI</h1>
          </div>
          
          <div className="flex items-center gap-3">
            <button 
              id="profile-toggle"
              className="px-5 py-2 bg-gray-100 text-gray-700 rounded-lg font-medium hover:bg-gray-200 transition-colors"
            >
              Profile
            </button>
            <form
              action={async () => {
                "use server"
                const supabase = await createClient()
                await supabase.auth.signOut()
                redirect("/")
              }}
            >
              <button className="px-5 py-2 bg-gray-100 text-gray-700 rounded-lg font-medium hover:bg-gray-200 transition-colors">
                Sign Out
              </button>
            </form>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="max-w-5xl mx-auto p-6 lg:p-8 relative">
        {/* Projects Section */}
        {regularProjects.length > 0 && (
          <div className="mb-12">
            <h2 className="text-3xl font-semibold text-gray-900 mb-6">Projects</h2>

            <div className="grid md:grid-cols-2 gap-6">
              {regularProjects.map((project: any) => (
                <ProjectCard 
                  key={project.id} 
                  project={project} 
                  progress={progressMap[project.id]} 
                  userLevel={userLevel}
                />
              ))}
            </div>
          </div>
        )}

        {/* Learning Modules Section */}
        {learningModules.length > 0 && (
          <div className="mb-12">
            <h2 className="text-3xl font-semibold text-gray-900 mb-6">Learning Modules</h2>

            <div className="grid md:grid-cols-2 gap-6">
              {learningModules.map((project: any) => (
                <ProjectCard 
                  key={project.id} 
                  project={project} 
                  progress={progressMap[project.id]}
                  userLevel={userLevel}
                />
              ))}
            </div>
          </div>
        )}

        {/* Empty State */}
        {learningModules.length === 0 && regularProjects.length === 0 && (
          <div className="bg-white rounded-xl p-12 text-center border border-gray-200">
            <div className="flex justify-center mb-6">
              <Image 
                src="/chameleon.png" 
                alt="Chameleon" 
                width={200} 
                height={200}
                className="opacity-80"
              />
            </div>
            <p className="text-gray-700 text-lg font-medium">
              No projects available yet.
            </p>
            <p className="text-gray-500 text-sm mt-2">
              Check back soon for exciting learning opportunities!
            </p>
          </div>
        )}
      </div>

      {/* Profile Sidebar (Hidden by default, shown when profile button clicked) */}
      <div 
        id="profile-sidebar"
        className="fixed top-0 right-0 h-full w-80 bg-white border-l border-gray-200 shadow-2xl transform translate-x-full transition-transform duration-300 z-50 overflow-y-auto"
      >
        <div className="p-6">
          <div className="flex justify-between items-center mb-6">
            <h2 className="text-xl font-semibold text-gray-900">Profile</h2>
            <button 
              id="profile-close"
              className="text-gray-400 hover:text-gray-900 transition-colors"
            >
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <line x1="18" y1="6" x2="6" y2="18"></line>
                <line x1="6" y1="6" x2="18" y2="18"></line>
              </svg>
            </button>
          </div>
          
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

      {/* Overlay */}
      <div 
        id="profile-overlay"
        className="fixed inset-0 bg-black/50 opacity-0 pointer-events-none transition-opacity duration-300 z-40"
      ></div>

      {/* Footer */}
      <div className="mt-16 border-t border-gray-200 bg-white relative">
        <div className="max-w-7xl mx-auto px-6 py-8">
          <div className="flex flex-col md:flex-row justify-between items-center gap-4">
            <p className="text-gray-500 text-sm">
              © 2025 CrekAI. Empowering learners through hands-on AI projects.
            </p>
            <div className="flex gap-6">
              <a href="#" className="text-gray-500 hover:text-gray-900 transition text-sm font-medium">
                About
              </a>
              <a href="#" className="text-gray-500 hover:text-gray-900 transition text-sm font-medium">
                Help
              </a>
              <a href="#" className="text-gray-500 hover:text-gray-900 transition text-sm font-medium">
                Contact
              </a>
            </div>
          </div>
        </div>
      </div>

      {/* Client-side Script for Sidebar Toggle */}
      <script dangerouslySetInnerHTML={{
        __html: `
          document.addEventListener('DOMContentLoaded', function() {
            const profileToggle = document.getElementById('profile-toggle');
            const profileClose = document.getElementById('profile-close');
            const profileSidebar = document.getElementById('profile-sidebar');
            const profileOverlay = document.getElementById('profile-overlay');

            function openSidebar() {
              profileSidebar.classList.remove('translate-x-full');
              profileOverlay.classList.remove('opacity-0', 'pointer-events-none');
            }

            function closeSidebar() {
              profileSidebar.classList.add('translate-x-full');
              profileOverlay.classList.add('opacity-0', 'pointer-events-none');
            }

            profileToggle.addEventListener('click', openSidebar);
            profileClose.addEventListener('click', closeSidebar);
            profileOverlay.addEventListener('click', closeSidebar);
          });
        `
      }} />
    </div>
  )
}

import { createClient } from "@/lib/supabase/server"
import { redirect } from "next/navigation"
import StepViewer from "@/components/step-viewer"

export default async function ProjectPage({
  params,
  searchParams,
}: {
  params: Promise<{ slug: string }>
  searchParams: Promise<{ step?: string }>
}) {
  const { slug } = await params
  const { step } = await searchParams
  const supabase = await createClient()
  const {
    data: { user },
  } = await supabase.auth.getUser()

  if (!user) {
    redirect("/auth/login")
  }

  // Fetch project by slug
  const { data: project, error: projectError } = await supabase.from("projects").select("*").eq("slug", slug).single()

  if (projectError || !project) {
    redirect("/projects")
  }

  // Fetch user progress
  const { data: progress, error: progressError } = await supabase
    .from("user_progress")
    .select("*")
    .eq("user_id", user.id)
    .eq("project_id", project.id)
    .maybeSingle()

  const currentStep = step ? Number.parseInt(step) : progress?.current_step || 1

  return <StepViewer project={project} progress={progress} currentStep={currentStep} userId={user.id} />
}

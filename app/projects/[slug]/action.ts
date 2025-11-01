"use server"

import { createClient } from "@/lib/supabase/server"

export async function resetProjectProgress(projectId: string) {
  const supabase = await createClient()

  const {
    data: { user },
  } = await supabase.auth.getUser()

  if (!user) {
    throw new Error("User not authenticated")
  }

  const { error } = await supabase.from("user_progress").delete().eq("user_id", user.id).eq("project_id", projectId)

  if (error) {
    throw new Error(`Failed to reset progress: ${error.message}`)
  }

  return { success: true }
}

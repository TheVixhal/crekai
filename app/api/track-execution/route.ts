import { createClient } from "@/lib/supabase/server"
import { NextRequest, NextResponse } from "next/server"

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { token, project_id, step } = body

    // Validate required fields
    if (!token || !project_id || !step) {
      return NextResponse.json(
        { error: "Missing required fields: token, project_id, step" },
        { status: 400 }
      )
    }

    const supabase = await createClient()

    // Find user by token
    const { data: profile, error: profileError } = await supabase
      .from("user_profiles")
      .select("id")
      .eq("colab_token", token)
      .single()

    if (profileError || !profile) {
      return NextResponse.json(
        { error: "Invalid token" },
        { status: 401 }
      )
    }

    const userId = profile.id

    // Find project by project_id (assuming it's the slug)
    const { data: project, error: projectError } = await supabase
      .from("projects")
      .select("id, total_steps")
      .eq("slug", project_id)
      .single()

    if (projectError || !project) {
      return NextResponse.json(
        { error: "Project not found" },
        { status: 404 }
      )
    }

    // Validate step number
    const stepNumber = Number.parseInt(step)
    if (isNaN(stepNumber) || stepNumber < 1 || stepNumber > project.total_steps) {
      return NextResponse.json(
        { error: "Invalid step number" },
        { status: 400 }
      )
    }

    // Get existing progress
    const { data: existingProgress } = await supabase
      .from("user_progress")
      .select("*")
      .eq("user_id", userId)
      .eq("project_id", project.id)
      .maybeSingle()

    const completedSteps = existingProgress?.completed_steps || []

    // Add step to completed if not already there
    if (!completedSteps.includes(stepNumber)) {
      completedSteps.push(stepNumber)
    }

    const nextStep = stepNumber + 1
    const hasNextStep = nextStep <= project.total_steps

    if (!existingProgress) {
      // Create new progress record
      const { error: insertError } = await supabase.from("user_progress").insert({
        user_id: userId,
        project_id: project.id,
        current_step: hasNextStep ? nextStep : stepNumber,
        completed_steps: completedSteps,
      })

      if (insertError) {
        console.error("Insert error:", insertError)
        return NextResponse.json(
          { error: "Failed to save progress" },
          { status: 500 }
        )
      }
    } else {
      // Update existing progress
      const { error: updateError } = await supabase
        .from("user_progress")
        .update({
          current_step: hasNextStep ? nextStep : stepNumber,
          completed_steps: completedSteps,
          last_accessed: new Date().toISOString(),
        })
        .eq("user_id", userId)
        .eq("project_id", project.id)

      if (updateError) {
        console.error("Update error:", updateError)
        return NextResponse.json(
          { error: "Failed to update progress" },
          { status: 500 }
        )
      }
    }

    return NextResponse.json({
      success: true,
      message: `Step ${stepNumber} completed successfully!`,
      next_step: hasNextStep ? nextStep : null,
      completed: true,
    })
  } catch (error) {
    console.error("Track execution error:", error)
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    )
  }
}

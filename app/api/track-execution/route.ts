import { createClient } from "@/lib/supabase/server"
import { NextRequest, NextResponse } from "next/server"
import fs from "fs/promises"
import path from "path"

// Load project configuration
async function loadProjectConfig(projectId: string) {
  try {
    const configPath = path.join(process.cwd(), "public", "projects", projectId, "config.json")
    const configData = await fs.readFile(configPath, "utf-8")
    return JSON.parse(configData)
  } catch (error) {
    console.error("Config load error:", error)
    return null
  }
}

// Validate code output against expected results
function validateOutput(output: any, expected: any): { valid: boolean; message: string } {
  try {
    // If no validation required, pass
    if (!expected) {
      return { valid: true, message: "No validation required" }
    }

    const { expected_variables } = expected

    if (!expected_variables) {
      return { valid: true, message: "No variable validation required" }
    }

    // Check each expected variable
    for (const [varName, varExpected] of Object.entries(expected_variables)) {
      const varData: any = varExpected

      // Check if variable exists in output
      if (!output.variables || !(varName in output.variables)) {
        return {
          valid: false,
          message: `Missing variable: ${varName}`,
        }
      }

      const varValue = output.variables[varName]

      // Type validation
      if (varData.type && varValue.type !== varData.type) {
        return {
          valid: false,
          message: `Variable '${varName}' has wrong type. Expected ${varData.type}, got ${varValue.type}`,
        }
      }

      // Shape validation (for arrays/tensors)
      if (varData.shape && JSON.stringify(varValue.shape) !== JSON.stringify(varData.shape)) {
        return {
          valid: false,
          message: `Variable '${varName}' has wrong shape. Expected ${JSON.stringify(varData.shape)}, got ${JSON.stringify(varValue.shape)}`,
        }
      }

      // Value validation
      if (varData.expected !== undefined && varValue.value !== varData.expected) {
        return {
          valid: false,
          message: `Variable '${varName}' has wrong value. Expected ${varData.expected}, got ${varValue.value}`,
        }
      }

      // Range validation
      if (varData.min_value !== undefined && varValue.min < varData.min_value) {
        return {
          valid: false,
          message: `Variable '${varName}' minimum value too low`,
        }
      }

      if (varData.max_value !== undefined && varValue.max > varData.max_value) {
        return {
          valid: false,
          message: `Variable '${varName}' maximum value too high`,
        }
      }
    }

    return { valid: true, message: "All validations passed" }
  } catch (error) {
    console.error("Validation error:", error)
    return { valid: false, message: "Validation failed" }
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { token, project_id, step, code, output } = body

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

    // Load project configuration
    const config = await loadProjectConfig(project_id)
    
    if (!config) {
      return NextResponse.json(
        { error: "Project configuration not found" },
        { status: 500 }
      )
    }

    // Find step configuration
    const stepConfig = config.steps.find((s: any) => s.step === stepNumber)

    if (!stepConfig) {
      return NextResponse.json(
        { error: "Step configuration not found" },
        { status: 400 }
      )
    }

    // Check if step has assignment
    if (stepConfig.has_assignment) {
      // Validate code and output are provided
      if (!code || !output) {
        return NextResponse.json(
          { error: "Code and output are required for assignment steps" },
          { status: 400 }
        )
      }

      // Validate output against expected results
      const validation = validateOutput(output, stepConfig.validation)

      if (!validation.valid) {
        return NextResponse.json(
          { 
            error: "Output validation failed", 
            message: validation.message,
            success: false 
          },
          { status: 400 }
        )
      }
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

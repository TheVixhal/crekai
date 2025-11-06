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

// Test-based validation
function validateWithTests(output: any, tests: any[]): { valid: boolean; message: string } {
  try {
    const variables = output.variables || {}
    const allVars = Object.values(variables) as any[]

    for (const test of tests) {
      const check = test.check

      // Array validation
      if (check.find_any_array) {
        const criteria = check.find_any_array
        let found = false

        for (const varData of allVars) {
          if (varData.type !== "numpy.ndarray" && varData.type !== "torch.Tensor") continue
          let matches = true

          if (criteria.shape && JSON.stringify(varData.shape) !== JSON.stringify(criteria.shape)) matches = false
          if (criteria.min !== undefined && Math.abs(varData.min - criteria.min) > 0.01) matches = false
          if (criteria.max !== undefined && Math.abs(varData.max - criteria.max) > 0.01) matches = false

          if (matches) { found = true; break }
        }

        if (!found) {
          const desc = test.description || test.name || "Array check"
          return {
            valid: false,
            message: `${desc} - Expected array with shape ${JSON.stringify(criteria.shape)}, min=${criteria.min}, max=${criteria.max}`
          }
        }
      }

      // Number validation
      if (check.find_any_number) {
        const criteria = check.find_any_number
        let found = false

        for (const varData of allVars) {
          if (varData.type !== "float" && varData.type !== "int") continue
          const tolerance = criteria.tolerance || 0.01
          if (Math.abs(varData.value - criteria.value) <= tolerance) { found = true; break }
        }

        if (!found) {
          const desc = test.description || test.name || "Number check"
          return { valid: false, message: `${desc} - Expected a number close to ${criteria.value}` }
        }
      }
    }

    return { valid: true, message: "All tests passed! Your logic is correct." }
  } catch (error) {
    console.error("Test validation error:", error)
    return { valid: false, message: "Validation failed" }
  }
}

function validateOutput(output: any, validation: any): { valid: boolean; message: string } {
  try {
    if (!validation) return { valid: true, message: "No validation required" }
    if (validation.type === "output_tests" && validation.tests)
      return validateWithTests(output, validation.tests)
    return { valid: true, message: "All validations passed" }
  } catch (error) {
    console.error("Validation error:", error)
    return { valid: false, message: "Validation failed" }
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    let { token, project_id, step, code, output } = body
    token = token?.trim()

    const supabase = await createClient()

    // ===== Validate token =====
    const { data: profile, error: profileError } = await supabase
      .from("user_profiles")
      .select("id")
      .eq("colab_token", token)
      .single()

    if (profileError || !profile)
      return NextResponse.json({ error: "Invalid token" }, { status: 401 })

    const userId = profile.id

    // ===== Validate project =====
    const { data: project, error: projectError } = await supabase
      .from("projects")
      .select("id, total_steps")
      .eq("slug", project_id)
      .single()

    if (projectError || !project)
      return NextResponse.json({ error: "Project not found" }, { status: 404 })

    const stepNumber = Number.parseInt(step)
    if (isNaN(stepNumber) || stepNumber < 1 || stepNumber > project.total_steps)
      return NextResponse.json({ error: "Invalid step number" }, { status: 400 })

    // ===== Load Config =====
    const config = await loadProjectConfig(project_id)
    if (!config)
      return NextResponse.json({ error: "Missing config.json for this project" }, { status: 500 })

    const stepConfig = config?.steps?.find((s: any) => s.step === stepNumber)

    // ===== Fetch existing progress =====
    const { data: existingProgress } = await supabase
      .from("user_progress")
      .select("*")
      .eq("user_id", userId)
      .eq("project_id", project.id)
      .maybeSingle()

    const completedSteps = existingProgress?.completed_steps || []
    const alreadyCompleted = completedSteps.includes(stepNumber)

    // ===== Re-verification mode =====
    if (alreadyCompleted) {
      console.log(`Step ${stepNumber} already completed - re-verification mode`)

      // Validate only against config (no DB writes)
      if (config && stepConfig?.has_assignment && code && output) {
        const validation = validateOutput(output, stepConfig.validation)
        if (!validation.valid) {
          return NextResponse.json({
            error: "Re-verification failed",
            message: validation.message,
            success: false,
            already_completed: true
          }, { status: 400 })
        }
      }

      // ✅ Re-verification passed, no DB changes
      return NextResponse.json({
        success: true,
        message: `Step ${stepNumber} re-verified successfully!`,
        already_completed: true
      })
    }

    // ===== First-time completion =====
    if (config && stepConfig?.has_assignment) {
      if (!code || !output)
        return NextResponse.json({ error: "Code and output required" }, { status: 400 })

      const validation = validateOutput(output, stepConfig.validation)
      if (!validation.valid)
        return NextResponse.json({
          error: "Assignment validation failed",
          message: validation.message,
          success: false
        }, { status: 400 })
    }

    // Mark as completed (DB write only once)
    const newSteps = Array.isArray(completedSteps) ? [...completedSteps, stepNumber] : [stepNumber]
    const nextStep = stepNumber + 1
    const hasNextStep = nextStep <= project.total_steps

    if (!existingProgress) {
      await supabase.from("user_progress").insert({
        user_id: userId,
        project_id: project.id,
        current_step: hasNextStep ? nextStep : stepNumber,
        completed_steps: newSteps,
      })
    } else {
      await supabase
        .from("user_progress")
        .update({
          current_step: hasNextStep ? nextStep : stepNumber,
          completed_steps: newSteps,
          last_accessed: new Date().toISOString(),
        })
        .eq("user_id", userId)
        .eq("project_id", project.id)
    }

    // ✅ First-time pass
    return NextResponse.json({
      success: true,
      message: `Step ${stepNumber} completed successfully!`,
      next_step: hasNextStep ? nextStep : null,
      already_completed: false
    })
  } catch (error) {
    console.error("Track execution error:", error)
    return NextResponse.json({ error: "Internal server error" }, { status: 500 })
  }
}

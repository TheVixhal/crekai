import { createClient } from "@/lib/supabase/server"
import { NextRequest, NextResponse } from "next/server"
import fs from "fs/promises"
import path from "path"

// ===== Load Project Config =====
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

// ===== Validation Logic =====
function validateWithTests(output: any, tests: any[]): { valid: boolean; message: string } {
  try {
    const variables = output?.variables || {}
    const allVars = Object.values(variables) as any[]

    for (const test of tests) {
      const check = test.check
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
          return {
            valid: false,
            message: `${test.description || "Array check"} - Expected shape ${JSON.stringify(criteria.shape)}, min=${criteria.min}, max=${criteria.max}`
          }
        }
      }

      if (check.find_any_number) {
        const criteria = check.find_any_number
        let found = false
        for (const varData of allVars) {
          if (varData.type !== "float" && varData.type !== "int") continue
          const tolerance = criteria.tolerance || 0.01
          if (Math.abs(varData.value - criteria.value) <= tolerance) {
            found = true
            break
          }
        }
        if (!found)
          return { valid: false, message: `${test.description || "Number check"} - Expected value near ${criteria.value}` }
      }
    }

    return { valid: true, message: "✅ All tests passed! Logic is correct." }
  } catch (err) {
    console.error("validateWithTests error:", err)
    return { valid: false, message: "Validation failed due to internal error." }
  }
}

function validateOutput(output: any, validation: any): { valid: boolean; message: string } {
  try {
    if (!validation) return { valid: true, message: "No validation required" }
    if (validation.type === "output_tests" && validation.tests)
      return validateWithTests(output, validation.tests)
    return { valid: true, message: "All validations passed" }
  } catch (err) {
    console.error("validateOutput error:", err)
    return { valid: false, message: "Validation failed" }
  }
}

// ===== MAIN HANDLER =====
export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    let { token, project_id, step, code, output } = body
    token = token?.trim()

    // If no token or project, reject
    if (!token || !project_id || !step) {
      return NextResponse.json({ error: "Missing required fields: token, project_id, step" }, { status: 400 })
    }

    // Load config.json for validation
    const config = await loadProjectConfig(project_id)
    if (!config) {
      return NextResponse.json({ error: "Missing config.json for this project" }, { status: 500 })
    }

    const stepNumber = parseInt(step)
    const stepConfig = config.steps?.find((s: any) => s.step === stepNumber)

    // Detect reverification mode — student already completed step
    const alreadyCompleted = !!output?.variables?.already_completed

    // ===== PURE CONFIG VALIDATION MODE =====
    if (alreadyCompleted) {
      console.log(`🔁 Re-verification for project=${project_id}, step=${stepNumber}`)
      if (!stepConfig?.validation) {
        return NextResponse.json({ success: true, message: "No validation configured." })
      }

      const validation = validateOutput(output, stepConfig.validation)
      if (!validation.valid) {
        return NextResponse.json({
          error: "Re-verification failed",
          message: validation.message,
          success: false,
          already_completed: true,
        }, { status: 400 })
      }

      return NextResponse.json({
        success: true,
        message: `Step ${stepNumber} re-verified successfully via config!`,
        already_completed: true,
      })
    }

    // ===== NORMAL VERIFICATION (FIRST ATTEMPT) =====
    console.log("🧠 Normal verification (first-time grading)")

    const supabase = await createClient()

    // 1️⃣ Validate token
    const { data: profile, error: profileError } = await supabase
      .from("user_profiles")
      .select("id")
      .eq("colab_token", token)
      .single()

    if (profileError || !profile) {
      return NextResponse.json({ error: "Invalid token" }, { status: 401 })
    }

    const userId = profile.id

    // 2️⃣ Get project
    const { data: project, error: projectError } = await supabase
      .from("projects")
      .select("id, total_steps")
      .eq("slug", project_id)
      .single()

    if (projectError || !project) {
      return NextResponse.json({ error: "Project not found" }, { status: 404 })
    }

    // 3️⃣ Validate step result using config.json
    if (stepConfig?.has_assignment && stepConfig.validation) {
      const validation = validateOutput(output, stepConfig.validation)
      if (!validation.valid) {
        return NextResponse.json({
          error: "Assignment validation failed",
          message: validation.message,
          success: false
        }, { status: 400 })
      }
    }

    // 4️⃣ Update DB progress (normal path)
    const nextStep = stepNumber + 1
    const hasNextStep = nextStep <= project.total_steps

    const { data: existingProgress } = await supabase
      .from("user_progress")
      .select("*")
      .eq("user_id", userId)
      .eq("project_id", project.id)
      .maybeSingle()

    const completedSteps = existingProgress?.completed_steps || []
    const newSteps = Array.from(new Set([...completedSteps, stepNumber]))

    if (existingProgress) {
      await supabase
        .from("user_progress")
        .update({
          current_step: hasNextStep ? nextStep : stepNumber,
          completed_steps: newSteps,
          last_accessed: new Date().toISOString(),
        })
        .eq("user_id", userId)
        .eq("project_id", project.id)
    } else {
      await supabase
        .from("user_progress")
        .insert({
          user_id: userId,
          project_id: project.id,
          current_step: hasNextStep ? nextStep : stepNumber,
          completed_steps: newSteps,
        })
    }

    // 5️⃣ Done
    return NextResponse.json({
      success: true,
      message: `Step ${stepNumber} completed successfully!`,
      next_step: hasNextStep ? nextStep : null,
    })
  } catch (err) {
    console.error("Track execution error:", err)
    return NextResponse.json({ error: "Internal server error", details: String(err) }, { status: 500 })
  }
}

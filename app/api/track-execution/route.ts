import { createClient } from "@/lib/supabase/server"
import { NextRequest, NextResponse } from "next/server"
import fs from "fs/promises"
import path from "path"

// ===== Load project config =====
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

// ===== Test-based validator =====
function validateWithTests(output: any, tests: any[]): { valid: boolean; message: string } {
  try {
    const variables = output.variables || {}
    const allVars = Object.values(variables) as any[]

    for (const test of tests) {
      const check = test.check

      // Array test
      if (check.find_any_array) {
        const criteria = check.find_any_array
        let found = false

        for (const varData of allVars) {
          if (varData.type !== "numpy.ndarray" && varData.type !== "torch.Tensor") continue

          let matches = true
          if (criteria.shape && JSON.stringify(varData.shape) !== JSON.stringify(criteria.shape)) matches = false
          if (criteria.min !== undefined && Math.abs(varData.min - criteria.min) > 0.01) matches = false
          if (criteria.max !== undefined && Math.abs(varData.max - criteria.max) > 0.01) matches = false

          if (matches) {
            found = true
            break
          }
        }

        if (!found) {
          const desc = test.description || test.name || "Array check"
          return {
            valid: false,
            message: `${desc} - Expected array with shape ${JSON.stringify(criteria.shape)}, min=${criteria.min}, max=${criteria.max}`
          }
        }
      }

      // Number test
      if (check.find_any_number) {
        const criteria = check.find_any_number
        if (criteria.value === undefined) {
          return { valid: false, message: "Test misconfigured: missing 'value' field in criteria" }
        }

        let found = false
        for (const varData of allVars) {
          if (varData.type !== "float" && varData.type !== "int") continue
          const tolerance = criteria.tolerance || 0.01
          if (Math.abs(varData.value - criteria.value) <= tolerance) {
            found = true
            break
          }
        }

        if (!found) {
          const desc = test.description || test.name || "Number check"
          return {
            valid: false,
            message: `${desc} - Expected a number close to ${criteria.value}`
          }
        }
      }
    }

    return { valid: true, message: "All tests passed! Your logic is correct." }
  } catch (error) {
    console.error("Test validation error:", error)
    return { valid: false, message: "Validation failed" }
  }
}

// ===== Output validation =====
function validateOutput(output: any, validation: any): { valid: boolean; message: string } {
  try {
    if (!validation) return { valid: true, message: "No validation required" }

    if (validation.type === "output_tests" && validation.tests) {
      return validateWithTests(output, validation.tests)
    }

    const { expected_variables } = validation
    if (!expected_variables) return { valid: true, message: "No variable validation required" }

    for (const [varName, varExpected] of Object.entries(expected_variables)) {
      const varData: any = varExpected
      if (!output.variables || !(varName in output.variables)) {
        return { valid: false, message: `Missing variable: ${varName}` }
      }

      const varValue = output.variables[varName]
      if (varData.type && varValue.type !== varData.type)
        return { valid: false, message: `Variable '${varName}' has wrong type. Expected ${varData.type}, got ${varValue.type}` }

      if (varData.shape && JSON.stringify(varValue.shape) !== JSON.stringify(varData.shape))
        return { valid: false, message: `Variable '${varName}' has wrong shape. Expected ${JSON.stringify(varData.shape)}, got ${JSON.stringify(varValue.shape)}` }

      if (varData.expected !== undefined && varValue.value !== varData.expected)
        return { valid: false, message: `Variable '${varName}' has wrong value. Expected ${varData.expected}, got ${varValue.value}` }

      if (varData.min_value !== undefined && varValue.min < varData.min_value)
        return { valid: false, message: `Variable '${varName}' minimum value too low` }

      if (varData.max_value !== undefined && varValue.max > varData.max_value)
        return { valid: false, message: `Variable '${varName}' maximum value too high` }
    }

    return { valid: true, message: "All validations passed" }
  } catch (error) {
    console.error("Validation error:", error)
    return { valid: false, message: "Validation failed" }
  }
}

// ===== Main Handler =====
export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    let { token, project_id, step, code, output } = body
    token = token?.trim()

    if (!token || !project_id || !step) {
      return NextResponse.json({ error: "Missing required fields: token, project_id, step" }, { status: 400 })
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
        { error: "Invalid token", details: "Token not found. Please regenerate." },
        { status: 401 }
      )
    }

    const userId = profile.id

    // Find project
    const { data: project, error: projectError } = await supabase
      .from("projects")
      .select("id, total_steps")
      .eq("slug", project_id)
      .single()

    if (projectError || !project)
      return NextResponse.json({ error: "Project not found" }, { status: 404 })

    const stepNumber = parseInt(step)
    if (isNaN(stepNumber) || stepNumber < 1 || stepNumber > project.total_steps)
      return NextResponse.json({ error: "Invalid step number" }, { status: 400 })

    // Load config
    const config = await loadProjectConfig(project_id)
    const stepConfig = config?.steps?.find((s: any) => s.step === stepNumber)

    // Get existing progress
    const { data: existingProgress } = await supabase
      .from("user_progress")
      .select("*")
      .eq("user_id", userId)
      .eq("project_id", project.id)
      .maybeSingle()

    const completedSteps = existingProgress?.completed_steps || []
    const alreadyCompleted = completedSteps.includes(stepNumber)

    // === Run Validation ===
    if (config && stepConfig?.has_assignment) {
      if (!code || !output) {
        return NextResponse.json({ error: "Code and output are required for assignment steps" }, { status: 400 })
      }

      const validation = validateOutput(output, stepConfig.validation)

      if (!validation.valid) {
        return NextResponse.json({
          error: "Assignment validation failed",
          message: validation.message,
          success: false,
          already_completed: alreadyCompleted,
        }, { status: 400 })
      }

      console.log("✅ Validation passed:", validation.message)
    }

    // === If already completed, just verify & return ===
    if (alreadyCompleted) {
      console.log(`Step ${stepNumber} already completed - skipping DB update.`)
      return NextResponse.json({
        success: true,
        message: `Step ${stepNumber} re-verified successfully!`,
        completed: true,
        already_completed: true,
        next_step: null,
      })
    }

    // === New step completion ===
    completedSteps.push(stepNumber)
    const nextStep = stepNumber + 1
    const hasNextStep = nextStep <= project.total_steps

    if (!existingProgress) {
      const { error: insertError } = await supabase.from("user_progress").insert({
        user_id: userId,
        project_id: project.id,
        current_step: hasNextStep ? nextStep : stepNumber,
        completed_steps: completedSteps,
      })
      if (insertError)
        return NextResponse.json({ error: "Failed to save progress", details: insertError.message }, { status: 500 })
    } else {
      const { error: updateError } = await supabase
        .from("user_progress")
        .update({
          current_step: hasNextStep ? nextStep : stepNumber,
          completed_steps: completedSteps,
          last_accessed: new Date().toISOString(),
        })
        .eq("user_id", userId)
        .eq("project_id", project.id)
      if (updateError)
        return NextResponse.json({ error: "Failed to update progress", details: updateError.message }, { status: 500 })
    }

    return NextResponse.json({
      success: true,
      message: `Step ${stepNumber} completed successfully!`,
      next_step: hasNextStep ? nextStep : null,
      completed: true,
      already_completed: false,
    })
  } catch (error) {
    console.error("Track execution error:", error)
    return NextResponse.json({ error: "Internal server error" }, { status: 500 })
  }
}

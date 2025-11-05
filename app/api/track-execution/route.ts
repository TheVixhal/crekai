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

// Test-based validation - checks logic, not variable names
function validateWithTests(output: any, tests: any[]): { valid: boolean; message: string } {
  try {
    const variables = output.variables || {}
    const allVars = Object.values(variables) as any[]

    for (const test of tests) {
      const check = test.check

      // Find any array matching criteria
      if (check.find_any_array) {
        const criteria = check.find_any_array
        let found = false

        for (const varData of allVars) {
          if (varData.type !== "numpy.ndarray" && varData.type !== "torch.Tensor") {
            continue
          }

          let matches = true

          // Check shape
          if (criteria.shape) {
            if (JSON.stringify(varData.shape) !== JSON.stringify(criteria.shape)) {
              matches = false
            }
          }

          // Check min value
          if (criteria.min !== undefined) {
            if (Math.abs(varData.min - criteria.min) > 0.01) {
              matches = false
            }
          }

          // Check max value
          if (criteria.max !== undefined) {
            if (Math.abs(varData.max - criteria.max) > 0.01) {
              matches = false
            }
          }

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

      // Find any number matching criteria
      if (check.find_any_number) {
        const criteria = check.find_any_number
        let found = false

        for (const varData of allVars) {
          if (varData.type !== "float" && varData.type !== "int") {
            continue
          }

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

// Validate code output against expected results
function validateOutput(output: any, validation: any): { valid: boolean; message: string } {
  try {
    // If no validation required, pass
    if (!validation) {
      return { valid: true, message: "No validation required" }
    }

    // New test-based validation (checks logic, not variable names)
    if (validation.type === "output_tests" && validation.tests) {
      return validateWithTests(output, validation.tests)
    }

    // Legacy validation (for backward compatibility)
    const { expected_variables } = validation

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
    let { token, project_id, step, code, output } = body

    // Trim token to remove any whitespace
    token = token?.trim()

    console.log("Track execution request:", { 
      tokenLength: token?.length,
      tokenStart: token?.substring(0, 10),
      tokenEnd: token?.substring(token.length - 10),
      project_id, 
      step,
      hasCode: !!code,
      hasOutput: !!output
    })

    // Validate required fields
    if (!token || !project_id || !step) {
      return NextResponse.json(
        { error: "Missing required fields: token, project_id, step" },
        { status: 400 }
      )
    }

    const supabase = await createClient()

    // Find user by token (RLS policy allows token lookup)
    console.log("Looking up token in database...")
    
    const { data: profile, error: profileError } = await supabase
      .from("user_profiles")
      .select("id")
      .eq("colab_token", token)
      .single()

    console.log("Token lookup result:", { 
      found: !!profile, 
      profileId: profile?.id,
      error: profileError?.message,
      errorCode: profileError?.code
    })

    if (profileError || !profile) {
      return NextResponse.json(
        { 
          error: "Invalid token",
          details: "Token not found in database. Please regenerate your token.",
          debug: {
            message: profileError?.message,
            code: profileError?.code,
            tokenLength: token.length
          }
        },
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
      console.log("No config found for project, allowing without validation")
      // No config = allow completion without validation (backward compatible)
    }

    // Find step configuration
    const stepConfig = config?.steps?.find((s: any) => s.step === stepNumber)

    // Check if step has assignment and validation
    if (config && stepConfig?.has_assignment) {
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
            error: "Assignment validation failed", 
            message: validation.message,
            success: false 
          },
          { status: 400 }
        )
      }

      console.log("✅ Validation passed:", validation.message)
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
      console.log("Creating new progress record for user:", userId)
      const { error: insertError } = await supabase.from("user_progress").insert({
        user_id: userId,
        project_id: project.id,
        current_step: hasNextStep ? nextStep : stepNumber,
        completed_steps: completedSteps,
      })

      if (insertError) {
        console.error("Insert error:", insertError)
        return NextResponse.json(
          { 
            error: "Failed to save progress",
            details: insertError.message,
            code: insertError.code
          },
          { status: 500 }
        )
      }
    } else {
      // Update existing progress
      console.log("Updating progress for user:", userId)
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
          { 
            error: "Failed to update progress",
            details: updateError.message,
            code: updateError.code
          },
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

import { NextRequest, NextResponse } from "next/server"
import fs from "fs/promises"
import path from "path"

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ slug: string }> }
) {
  try {
    const { slug } = await params
    const { searchParams } = new URL(request.url)
    const step = searchParams.get("step")

    if (!step) {
      return NextResponse.json(
        { error: "Step parameter required" },
        { status: 400 }
      )
    }

    const stepNumber = Number.parseInt(step)

    // Load project config
    const configPath = path.join(process.cwd(), "public", "projects", slug, "config.json")
    
    try {
      const configData = await fs.readFile(configPath, "utf-8")
      const config = JSON.parse(configData)
      
      // Find step config
      const stepConfig = config.steps.find((s: any) => s.step === stepNumber)
      
      if (!stepConfig) {
        return NextResponse.json({
          success: true,
          has_assignment: false,
          variables: []
        })
      }

      // Extract variable names from validation config
      const variables = stepConfig.validation?.expected_variables 
        ? Object.keys(stepConfig.validation.expected_variables)
        : []

      return NextResponse.json({
        success: true,
        has_assignment: stepConfig.has_assignment || false,
        variables,
        validation: stepConfig.validation || null
      })
    } catch (error) {
      // No config file = no validation required
      return NextResponse.json({
        success: true,
        has_assignment: false,
        variables: []
      })
    }
  } catch (error) {
    console.error("Get step info error:", error)
    return NextResponse.json(
      { error: "Failed to get step info" },
      { status: 500 }
    )
  }
}

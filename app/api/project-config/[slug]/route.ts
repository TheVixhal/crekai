import { NextRequest, NextResponse } from "next/server"
import fs from "fs/promises"
import path from "path"

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ slug: string }> }
) {
  try {
    const { slug } = await params
    
    const configPath = path.join(process.cwd(), "public", "projects", slug, "config.json")
    
    try {
      const configData = await fs.readFile(configPath, "utf-8")
      const config = JSON.parse(configData)
      
      return NextResponse.json({
        success: true,
        config,
      })
    } catch (error) {
      // Config file doesn't exist - return default (all steps have assignments)
      return NextResponse.json({
        success: true,
        config: null,
      })
    }
  } catch (error) {
    console.error("Get config error:", error)
    return NextResponse.json(
      { error: "Failed to load configuration" },
      { status: 500 }
    )
  }
}

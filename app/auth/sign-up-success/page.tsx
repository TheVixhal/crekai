import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"

export default function SignUpSuccessPage() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-slate-950 to-slate-900">
      <Card className="w-full max-w-md p-8 text-center bg-slate-800 border-slate-700">
        <div className="mb-6">
          <div className="w-16 h-16 bg-green-500/20 rounded-full mx-auto flex items-center justify-center mb-4">
            <svg className="w-8 h-8 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
            </svg>
          </div>
          <h1 className="text-2xl font-bold text-white mb-2">Check Your Email</h1>
          <p className="text-slate-300">
            We've sent you a confirmation email. Please verify your email to start learning!
          </p>
        </div>

        <Link href="/auth/login">
          <Button className="w-full bg-blue-600 hover:bg-blue-700">Back to Sign In</Button>
        </Link>
      </Card>
    </div>
  )
}

import Link from "next/link";
import { createClient } from "@/lib/supabase/server";
import { redirect } from "next/navigation";
import TypewriterHeading from "@/components/TypewriterHeading";
import CrackEffect from "@/components/CrackEffect";
import { FloatingNav } from "@/components/ui/floating-navbar";
import { IconHome, IconUser, IconMessage } from "@tabler/icons-react";

export default async function Home() {
  const navItems = [
    {
      name: "Home",
      link: "/",
      icon: <IconHome className="h-4 w-4 text-neutral-500 dark:text-white" />,
    },
    {
      name: "About",
      link: "/about",
      icon: <IconUser className="h-4 w-4 text-neutral-500 dark:text-white" />,
    },
    {
      name: "Contact",
      link: "/contact",
      icon: <IconMessage className="h-4 w-4 text-neutral-500 dark:text-white" />,
    },
  ];
  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();
  if (user) {
    redirect("/projects");
  }
  return (
    <CrackEffect>
      <div className="relative w-full">
        <FloatingNav navItems={navItems} />
        <div className="min-h-screen flex flex-col bg-gradient-to-br from-amber-50 via-orange-50 to-amber-100">
          <div className="w-full max-w-7xl mx-auto">
            {/* HERO CARD */}
            <div className="border-b-4 border-black py-12 px-6 text-center bg-gradient-to-b from-white to-gray-50 shadow-lg relative overflow-hidden rounded-b-3xl parallax-header">
              <div className="absolute inset-0 opacity-5 parallax-grid" style={{
                backgroundImage: `repeating-linear-gradient(0deg, #000 0px, #000 1px, transparent 1px, transparent 4px),
                                 repeating-linear-gradient(90deg, #000 0px, #000 1px, transparent 1px, transparent 4px)`
              }}></div>
              <div className="relative z-10">
                <h1 className="text-7xl md:text-8xl font-bold text-black font-serif tracking-wider mb-4 drop-shadow-sm">Crek AI</h1>
                <div className="h-2 bg-gradient-to-r from-transparent via-black to-transparent mx-auto w-64 mb-4 shadow-md"></div>
                <p className="text-lg md:text-xl text-gray-800 font-serif italic mb-2">
                  Learn AI & Machine Learning Through Hands On Practice
                </p>
                <p className="text-xs text-gray-600 tracking-widest font-sans uppercase">Your Source for AI/ML Mastery</p>
              </div>
            </div>
            {/* POPULAR PROJECTS - truly connected, no forced margin */}
            <div className="border-t-4 border-black bg-gradient-to-b from-white to-gray-50 shadow-inner">
              <div className="px-6 py-12">
                <h3 className="text-3xl font-bold font-serif text-black mb-8">Popular Projects</h3>
                <div className="grid md:grid-cols-3 gap-6">
                  <div className="border-2 border-black p-6 bg-white rounded">
                    <p className="text-xs uppercase tracking-widest font-bold mb-3 text-blue-600 font-sans relative z-10">Beginner</p>
                    <h4 className="text-xl font-bold font-serif mb-3 text-black relative z-10">Build Artificial Neural Network</h4>
                    <p className="text-sm text-gray-700 mb-4 relative z-10">
                      An artificial neural network (ANN) is a computational model inspired by the human brain's structure and function, consisting of interconnected nodes called neurons.
                    </p>
                    <p className="text-xs text-gray-600 font-sans relative z-10">15 Steps • Beginner Friendly</p>
                  </div>
                  <div className="border-2 border-black p-6 bg-white rounded">
                    <p className="text-xs uppercase tracking-widest font-bold mb-3 text-purple-600 font-sans relative z-10">Intermediate</p>
                    <h4 className="text-xl font-bold font-serif mb-3 text-black relative z-10">Build Tokeniser</h4>
                    <p className="text-sm text-gray-700 mb-4 relative z-10">
                      A tokenizer in Natural Language Processing (NLP) is a tool that breaks down raw text into smaller, manageable units called tokens.
                    </p>
                    <p className="text-xs text-gray-600 font-sans relative z-10">20 Steps • Intermediate Project</p>
                  </div>
                  <div className="border-2 border-black p-6 bg-white rounded">
                    <p className="text-xs uppercase tracking-widest font-bold mb-3 text-green-600 font-sans relative z-10">Advanced</p>
                    <h4 className="text-xl font-bold font-serif mb-3 text-black relative z-10">Build Your Own GPT</h4>
                    <p className="text-sm text-gray-700 mb-4 relative z-10">
                      GPT stands for Generative Pre-trained Transformer, which is a type of language model that uses a neural network architecture called a transformer to generate human-like text.
                    </p>
                    <p className="text-xs text-gray-600 font-sans relative z-10">25+ Steps • Advanced Project</p>
                  </div>
                </div>
                {/* FEATURED DEMO */}
                <div className="mt-8 border-2 border-black bg-white rounded-lg overflow-hidden shadow-xl hover:shadow-2xl transition-shadow duration-300">
                  <div className="border-b-2 border-black p-4 bg-gradient-to-r from-amber-50 to-orange-50">
                    <p className="text-xs uppercase tracking-widest font-sans font-bold mb-1 text-orange-600">Featured Demo</p>
                    <h4 className="text-xl font-bold font-serif text-black">See Our Projects in Action</h4>
                  </div>
                  <div className="p-6 bg-white">
                    <div className="relative w-full rounded overflow-hidden border-2 border-black" style={{ paddingBottom: '56.25%' }}>
                      <iframe
                        className="absolute top-0 left-0 w-full h-full"
                        src="https://www.youtube.com/embed/kmJz8w5ij8Y?si=jGULg0D88G6GFkfI"
                        title="CrekAI Project Demo"
                        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                        allowFullScreen
                      />
                    </div>
                  </div>
                </div>
              </div>
            </div>
            {/* TESTIMONIALS */}
            <div className="border-t-4 border-black bg-gradient-to-br from-amber-50 to-orange-50 py-16 px-6">
              <div className="max-w-7xl mx-auto">
                <h3 className="text-4xl font-bold font-serif text-black mb-4 text-center">Hear it from our members</h3>
                <p className="text-center text-gray-700 mb-12 font-sans">See what learners are saying about their CrekAI experience</p>
                <div className="grid md:grid-cols-3 gap-6">
                  <div className="bg-white border-2 border-black p-6 relative overflow-hidden rounded-lg shadow-lg hover:shadow-2xl transition-all">
                    <div className="absolute top-0 left-0 w-16 h-16 bg-orange-200 opacity-20 rounded-full blur-2xl"></div>
                    <div className="relative z-10">
                      <div className="flex items-center mb-4">
                        <div className="w-12 h-12 rounded-full bg-gradient-to-br from-orange-400 to-amber-400 flex items-center justify-center text-white font-bold text-xl border-2 border-black">
                          S
                        </div>
                        <div className="ml-3">
                          <p className="font-bold font-serif text-black">Sarah Chen</p>
                          <p className="text-xs text-gray-600 font-sans">ML Engineer at Google</p>
                        </div>
                      </div>
                      <p className="text-sm text-gray-700 font-sans leading-relaxed">
                        "CrekAI transformed how I learn AI. The hands-on approach helped me land my dream job at Google. The projects are practical and industry-relevant!"
                      </p>
                      <div className="mt-4 flex gap-1">
                        <span className="text-orange-500">★</span>
                        <span className="text-orange-500">★</span>
                        <span className="text-orange-500">★</span>
                        <span className="text-orange-500">★</span>
                        <span className="text-orange-500">★</span>
                      </div>
                    </div>
                  </div>
                  <div className="bg-white border-2 border-black p-6 relative overflow-hidden rounded-lg shadow-lg hover:shadow-2xl transition-all">
                    <div className="absolute top-0 left-0 w-16 h-16 bg-purple-200 opacity-20 rounded-full blur-2xl"></div>
                    <div className="relative z-10">
                      <div className="flex items-center mb-4">
                        <div className="w-12 h-12 rounded-full bg-gradient-to-br from-purple-400 to-pink-400 flex items-center justify-center text-white font-bold text-xl border-2 border-black">
                          M
                        </div>
                        <div className="ml-3">
                          <p className="font-bold font-serif text-black">Michael Rodriguez</p>
                          <p className="text-xs text-gray-600 font-sans">Data Scientist at Meta</p>
                        </div>
                      </div>
                      <p className="text-sm text-gray-700 font-sans leading-relaxed">
                        "Finally, a platform that focuses on doing rather than just watching. I built my GPT model from scratch and gained confidence in my abilities."
                      </p>
                      <div className="mt-4 flex gap-1">
                        <span className="text-orange-500">★</span>
                        <span className="text-orange-500">★</span>
                        <span className="text-orange-500">★</span>
                        <span className="text-orange-500">★</span>
                        <span className="text-orange-500">★</span>
                      </div>
                    </div>
                  </div>
                  <div className="bg-white border-2 border-black p-6 relative overflow-hidden rounded-lg shadow-lg hover:shadow-2xl transition-all">
                    <div className="absolute top-0 left-0 w-16 h-16 bg-blue-200 opacity-20 rounded-full blur-2xl"></div>
                    <div className="relative z-10">
                      <div className="flex items-center mb-4">
                        <div className="w-12 h-12 rounded-full bg-gradient-to-br from-blue-400 to-cyan-400 flex items-center justify-center text-white font-bold text-xl border-2 border-black">
                          P
                        </div>
                        <div className="ml-3">
                          <p className="font-bold font-serif text-black">Priya Sharma</p>
                          <p className="text-xs text-gray-600 font-sans">AI Researcher at Microsoft</p>
                        </div>
                      </div>
                      <p className="text-sm text-gray-700 font-sans leading-relaxed">
                        "The step-by-step projects made complex concepts crystal clear. CrekAI is the best investment I made in my AI career journey."
                      </p>
                      <div className="mt-4 flex gap-1">
                        <span className="text-orange-500">★</span>
                        <span className="text-orange-500">★</span>
                        <span className="text-orange-500">★</span>
                        <span className="text-orange-500">★</span>
                        <span className="text-orange-500">★</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
            {/* FAQ SECTION */}
            <div className="border-t-4 border-black bg-white py-16 px-6">
              <div className="max-w-4xl mx-auto">
                <h3 className="text-4xl font-bold font-serif text-black mb-4 text-center">Frequently Asked Questions</h3>
                <p className="text-center text-gray-700 mb-12 font-sans">Everything you need to know about CrekAI</p>
                <div className="space-y-4">
                  <details className="group border-2 border-black rounded-lg overflow-hidden bg-white hover:shadow-lg transition-shadow">
                    <summary className="cursor-pointer p-6 font-bold font-serif text-lg text-black bg-gradient-to-r from-gray-50 to-white hover:from-amber-50 hover:to-orange-50 transition-colors flex justify-between items-center">
                      What makes CrekAI different from other learning platforms?
                      <span className="text-2xl group-open:rotate-45 transition-transform">+</span>
                    </summary>
                    <div className="p-6 pt-0 text-gray-700 font-sans border-t-2 border-black">
                      <p>CrekAI focuses on hands-on, project-based learning rather than passive video watching. You'll build real AI projects from scratch, gaining practical experience that employers value. Our structured learning paths guide you from beginner to advanced concepts with clear milestones and achievements.</p>
                    </div>
                  </details>
                  <details className="group border-2 border-black rounded-lg overflow-hidden bg-white hover:shadow-lg transition-shadow">
                    <summary className="cursor-pointer p-6 font-bold font-serif text-lg text-black bg-gradient-to-r from-gray-50 to-white hover:from-amber-50 hover:to-orange-50 transition-colors flex justify-between items-center">
                      Do I need prior programming experience?
                      <span className="text-2xl group-open:rotate-45 transition-transform">+</span>
                    </summary>
                    <div className="p-6 pt-0 text-gray-700 font-sans border-t-2 border-black">
                      <p>While basic programming knowledge is helpful, our beginner projects are designed for those new to AI and ML. We provide step-by-step guidance and explain concepts clearly. If you're completely new to programming, we recommend starting with our foundational projects.</p>
                    </div>
                  </details>
                  <details className="group border-2 border-black rounded-lg overflow-hidden bg-white hover:shadow-lg transition-shadow">
                    <summary className="cursor-pointer p-6 font-bold font-serif text-lg text-black bg-gradient-to-r from-gray-50 to-white hover:from-amber-50 hover:to-orange-50 transition-colors flex justify-between items-center">
                      How long does it take to complete a project?
                      <span className="text-2xl group-open:rotate-45 transition-transform">+</span>
                    </summary>
                    <div className="p-6 pt-0 text-gray-700 font-sans border-t-2 border-black">
                      <p>Project duration varies based on complexity and your pace. Beginner projects typically take 2-4 hours, intermediate projects 5-8 hours, and advanced projects 10-15 hours. You can work at your own speed and pause/resume anytime.</p>
                    </div>
                  </details>
                  <details className="group border-2 border-black rounded-lg overflow-hidden bg-white hover:shadow-lg transition-shadow">
                    <summary className="cursor-pointer p-6 font-bold font-serif text-lg text-black bg-gradient-to-r from-gray-50 to-white hover:from-amber-50 hover:to-orange-50 transition-colors flex justify-between items-center">
                      Can I access projects after completing them?
                      <span className="text-2xl group-open:rotate-45 transition-transform">+</span>
                    </summary>
                    <div className="p-6 pt-0 text-gray-700 font-sans border-t-2 border-black">
                      <p>Yes! All your completed projects remain accessible in your account. You can revisit them anytime to review concepts, reference code, or build upon what you've learned. Your progress and achievements are saved permanently.</p>
                    </div>
                  </details>
                  <details className="group border-2 border-black rounded-lg overflow-hidden bg-white hover:shadow-lg transition-shadow">
                    <summary className="cursor-pointer p-6 font-bold font-serif text-lg text-black bg-gradient-to-r from-gray-50 to-white hover:from-amber-50 hover:to-orange-50 transition-colors flex justify-between items-center">
                      Is there a money-back guarantee?
                      <span className="text-2xl group-open:rotate-45 transition-transform">+</span>
                    </summary>
                    <div className="p-6 pt-0 text-gray-700 font-sans border-t-2 border-black">
                      <p>We offer a 30-day money-back guarantee. If you're not satisfied with CrekAI for any reason within the first 30 days, contact our support team for a full refund. No questions asked.</p>
                    </div>
                  </details>
                </div>
              </div>
            </div>
            {/* FOOTER */}
            <div className="border-t-4 border-black bg-gradient-to-t from-gray-100 to-white text-black py-8 px-6 text-center">
              <p className="font-serif text-sm mb-2">CrekAI - Master AI & Machine Learning</p>
              <p className="text-xs text-gray-500">© 2025 All rights reserved</p>
            </div>
          </div>
        </div>
      </div>
    </CrackEffect>
  );
}
